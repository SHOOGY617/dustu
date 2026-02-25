import cv2
import time
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# ==========================================
# --- 1. SYSTEM CONFIGURATION ---
# ==========================================
CAMERA_URL = 1  
MQTT_BROKER = "broker.emqx.io"
TOPIC_CONTROL = "dustu/control"
TOPIC_VOICE = "dustu/voice" # NEW: Listens for website commands!

# --- STATE MACHINE ---
current_mode = "IDLE"  
last_mqtt_time = 0
SORT_COOLDOWN = 6      
HEARTBEAT_RATE = 0.15  

target_person_id = None
needs_target_lock = False
summon_start_time = 0
summon_drive_duration = 0
return_start_time = 0

TRASH_MIN_AREA = 0.05  

CLASS_PERSON = 0
WET_ITEMS = [46, 47, 48, 49, 50, 51, 52, 53, 54, 55] 
DRY_ITEMS = [39, 40, 41, 42, 43, 44, 45, 67, 73]     

# ==========================================
# --- 2. CLOUD TELEPATHY (MQTT CALLBACK) ---
# ==========================================
def on_message(client, userdata, msg):
    global current_mode, needs_target_lock, target_person_id, summon_start_time
    
    # We only care about messages on the Voice topic
    if msg.topic == TOPIC_VOICE:
        command = msg.payload.decode().lower()
        print(f"☁️ CLOUD COMMAND RECEIVED: '{command}'")
        
        if "follow" in command:
            current_mode = "FOLLOW"
            needs_target_lock = True 
            print(">> MODE: FOLLOW ME")
            
        elif "come" in command or "here" in command:
            current_mode = "SUMMON"
            needs_target_lock = True
            summon_start_time = time.time()
            print(">> MODE: SUMMONED")

        elif "sort" in command or "trash" in command:
            current_mode = "SORT"
            print(">> MODE: SORTING")
            
        elif "stop" in command or "halt" in command:
            current_mode = "IDLE"
            client.publish(TOPIC_CONTROL, "S")
            print(">> MODE: IDLE")

print("🌐 CONNECTING TO CLOUD...")
client = mqtt.Client()
client.on_message = on_message # Attach the listener
client.connect(MQTT_BROKER, 1883, 60)
client.subscribe(TOPIC_VOICE)  # Subscribe to the voice channel
client.loop_start()

print("🧠 LOADING VISION CORTEX...")
model = YOLO("yolov8n.pt") 

# ==========================================
# --- 3. THE MAIN VISION LOOP ---
# ==========================================
cap = cv2.VideoCapture(CAMERA_URL)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_h, frame_w = frame.shape[:2]
    frame_center_x = frame_w // 2
    screen_area = frame_w * frame_h
    
    results = model.track(frame, persist=True, stream=True, verbose=False)
    current_time = time.time()
    
    too_close = False
    active_target_box = None

    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            track_id = int(box.id[0]) if box.id is not None else -1
            box_area = (x2 - x1) * (y2 - y1)
            area_ratio = box_area / screen_area

            # --- PIXEL PROXIMITY SENSOR ---
            if area_ratio > 0.60:
                too_close = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

            if conf > 0.5:
                color = (255, 255, 255) 

                # --- TRASH DETECTION LOGIC ---
                if (class_id in DRY_ITEMS or class_id in WET_ITEMS) and area_ratio > TRASH_MIN_AREA:
                    if current_mode == "WAITING_TRASH" or current_mode == "FOLLOW" or current_mode == "SORT":
                        client.publish(TOPIC_CONTROL, "S") 
                        
                        if class_id in DRY_ITEMS:
                            color = (0, 255, 255)
                            if (current_time - last_mqtt_time > SORT_COOLDOWN):
                                print(f"🧃 SORTING DRY: {name}")
                                client.publish(TOPIC_CONTROL, "D")
                                last_mqtt_time = current_time
                                if current_mode == "WAITING_TRASH":
                                    current_mode = "RETURNING"
                                    return_start_time = current_time + 4 
                                    
                        elif class_id in WET_ITEMS:
                            color = (255, 0, 0)
                            if (current_time - last_mqtt_time > SORT_COOLDOWN):
                                print(f"🍎 SORTING WET: {name}")
                                client.publish(TOPIC_CONTROL, "W")
                                last_mqtt_time = current_time
                                if current_mode == "WAITING_TRASH":
                                    current_mode = "RETURNING"
                                    return_start_time = current_time + 4

                # --- TARGET ACQUISITION ---
                if (current_mode == "FOLLOW" or current_mode == "SUMMON") and class_id == CLASS_PERSON:
                    person_center_x = (x1 + x2) // 2
                    if needs_target_lock and abs(person_center_x - frame_center_x) < 100:
                        target_person_id = track_id
                        needs_target_lock = False
                        
                    if track_id == target_person_id:
                        color = (0, 255, 0) 
                        active_target_box = (x1, y1, x2, y2)
                        cv2.putText(frame, f"LOCKED:{track_id}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        if current_mode == "SUMMON" and area_ratio > 0.30:
                            current_mode = "WAITING_TRASH"
                            client.publish(TOPIC_CONTROL, "S")

                if not too_close:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{name}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ==========================================
    # --- 6. MOVEMENT ENGINE ---
    # ==========================================
    if too_close and current_mode != "RETURNING":
        client.publish(TOPIC_CONTROL, "S")
        cv2.putText(frame, "PROXIMITY HALT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    elif (current_mode == "FOLLOW" or current_mode == "SUMMON") and active_target_box is not None:
        px1, py1, px2, py2 = active_target_box
        person_center_x = (px1 + px2) // 2
        
        if current_time - last_mqtt_time > HEARTBEAT_RATE:
            if person_center_x < (frame_center_x - 60): client.publish(TOPIC_CONTROL, "L")
            elif person_center_x > (frame_center_x + 60): client.publish(TOPIC_CONTROL, "R")
            else: client.publish(TOPIC_CONTROL, "F")
            last_mqtt_time = current_time

    # THE SAFE "HAPPY WIGGLE & SLEEP" LOGIC
    elif current_mode == "RETURNING":
        if current_time > return_start_time: 
            time_spent = current_time - return_start_time
            if time_spent < 0.5: client.publish(TOPIC_CONTROL, "L")
            elif time_spent < 1.0: client.publish(TOPIC_CONTROL, "R")
            else:
                client.publish(TOPIC_CONTROL, "S")
                current_mode = "IDLE"

    hud_color = (0, 255, 0) if current_mode in ["FOLLOW", "SUMMON"] else (0, 255, 255)
    cv2.putText(frame, f"MODE: {current_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)

    cv2.imshow("DUSTU_OS VISION", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

client.publish(TOPIC_CONTROL, "S")
cap.release()
cv2.destroyAllWindows()
