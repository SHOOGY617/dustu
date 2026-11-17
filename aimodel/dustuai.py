import cv2
import time
import sys
import argparse
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# ==============================================================================
# --- 1. SYSTEM CONFIGURATION & CAMERA SETUP ---
# ==============================================================================
# CLI Argument parsing allows custom camera index or network stream URL:
# python dustuai.py --cam 1          (OBS Virtual Camera, recommended)
# python dustuai.py --cam 0          (Default built-in webcam)
# python dustuai.py --cam "http://192.168.1.5:8080/video" (Phone IP Webcam)
parser = argparse.ArgumentParser(description="DUSTU AI Multi-Category Waste Vision Cortex")
parser.add_argument("--cam", default="1", help="Camera device index (1 for OBS Virtual Cam, 0 for Webcam) or stream URL")
args, _ = parser.parse_known_args()

try:
    CAMERA_SOURCE = int(args.cam)
except ValueError:
    CAMERA_SOURCE = args.cam

MQTT_BROKER = "broker.emqx.io"
TOPIC_CONTROL = "dustu/control"
TOPIC_VOICE = "dustu/voice"
TOPIC_TELEMETRY = "dustu/telemetry"

# --- STATE MACHINE ---
current_mode = "IDLE"  
last_mqtt_time = 0
SORT_COOLDOWN = 6      
HEARTBEAT_RATE = 0.15  

target_person_id = None
needs_target_lock = False
summon_start_time = 0
return_start_time = 0

TRASH_MIN_AREA = 0.04  # Minimum relative bounding box area to trigger sorting

# ==============================================================================
# --- 2. MULTI-CATEGORY WASTE CLASSIFICATION MATRIX (COCO-80) ---
# ==============================================================================
CLASS_PERSON = 0

# 1. ORGANIC / WET / COMPOSTABLE (Biodegradable Food & Flora)
WET_ITEMS = {
    46: "Banana",
    47: "Apple",
    48: "Sandwich",
    49: "Orange",
    50: "Broccoli",
    51: "Carrot",
    52: "Hot Dog",
    53: "Pizza",
    54: "Donut",
    55: "Cake",
    58: "Plant/Organic"
}

# 2. DRY / RECYCLABLE (Plastics, Packaging, Glass, Paper, Cans)
DRY_ITEMS = {
    39: "Bottle (Plastic/Glass)",
    40: "Wine Glass",
    41: "Cup/Can",
    42: "Fork",
    44: "Spoon",
    45: "Bowl",
    73: "Book/Paper",
    75: "Vase",
    79: "Toothbrush"
}

# 3. ELECTRONIC WASTE (High-Value / Toxic Recyclables)
EWASTE_ITEMS = {
    63: "Laptop",
    64: "Mouse",
    65: "Remote Control",
    66: "Keyboard",
    67: "Cell Phone",
    70: "Toaster",
    74: "Clock",
    78: "Hair Drier"
}

# 4. HAZARDOUS / SHARP (Safety Alert - Operator Attention)
HAZARD_ITEMS = {
    43: "Knife (Sharp)",
    76: "Scissors (Sharp)"
}

# 5. TEXTILE / PERSONAL GOODS (Protected - Do Not Pick Up)
TEXTILE_ITEMS = {
    24: "Backpack",
    26: "Handbag",
    27: "Tie"
}

def classify_object(class_id, default_name):
    """
    Categorizes YOLO detections into target bins, tags, and color codes:
    Returns dict: {cat, label, cmd, color, is_trash}
    """
    if class_id in WET_ITEMS:
        return {
            "cat": "WET",
            "tag": "WET/BIO",
            "name": WET_ITEMS[class_id],
            "cmd": "W",
            "color": (0, 230, 100),       # Vibrant Green (Compost)
            "is_trash": True
        }
    elif class_id in DRY_ITEMS:
        return {
            "cat": "DRY",
            "tag": "DRY/RECYCLE",
            "name": DRY_ITEMS[class_id],
            "cmd": "D",
            "color": (0, 235, 255),       # Yellow/Cyan (Recyclable)
            "is_trash": True
        }
    elif class_id in EWASTE_ITEMS:
        # Route to Dry bin away from wet organic decomposition; send telemetry
        return {
            "cat": "EWASTE",
            "tag": "E-WASTE",
            "name": EWASTE_ITEMS[class_id],
            "cmd": "D",
            "color": (255, 0, 255),       # Neon Magenta
            "is_trash": True
        }
    elif class_id in HAZARD_ITEMS:
        return {
            "cat": "HAZARD",
            "tag": "⚠️ HAZARD/SHARP",
            "name": HAZARD_ITEMS[class_id],
            "cmd": "S",
            "color": (0, 0, 255),         # Alert Red
            "is_trash": False
        }
    elif class_id in TEXTILE_ITEMS:
        return {
            "cat": "TEXTILE",
            "tag": "PERSONAL ASSET",
            "name": TEXTILE_ITEMS[class_id],
            "cmd": None,
            "color": (220, 100, 255),     # Violet
            "is_trash": False
        }
    return None

# ==============================================================================
# --- 3. CLOUD TELEPATHY (MQTT CONTROLLER) ---
# ==============================================================================
def on_message(client, userdata, msg):
    global current_mode, needs_target_lock, target_person_id, summon_start_time
    
    if msg.topic == TOPIC_VOICE:
        command = msg.payload.decode().strip().lower()
        print(f"☁️ CLOUD VOICE COMMAND: '{command}'")
        
        if "follow" in command:
            current_mode = "FOLLOW"
            needs_target_lock = True 
            print(">> MODE SET: [FOLLOW ME]")
            
        elif "come" in command or "here" in command or "summon" in command:
            current_mode = "SUMMON"
            needs_target_lock = True
            summon_start_time = time.time()
            print(">> MODE SET: [SUMMON TO TARGET]")

        elif "sort" in command or "trash" in command:
            current_mode = "SORT"
            print(">> MODE SET: [AUTONOMOUS SORTING]")
            
        elif "stop" in command or "halt" in command or "idle" in command:
            current_mode = "IDLE"
            client.publish(TOPIC_CONTROL, "S")
            print(">> MODE SET: [IDLE / STANDBY]")

print("🌐 CONNECTING TO MQTT CLOUD BROKER...")
client = mqtt.Client()
client.on_message = on_message
try:
    client.connect(MQTT_BROKER, 1883, 60)
    client.subscribe(TOPIC_VOICE)
    client.loop_start()
    print("✅ CLOUD MQTT CONNECTED!")
except Exception as e:
    print(f"⚠️ MQTT Warning: {e}. Running in local perception mode.")

print("🧠 LOADING YOLOV8 VISION CORTEX...")
model = YOLO("yolov8n.pt") 

# ==============================================================================
# --- 4. VIDEO STREAM INITIALIZATION ---
# ==============================================================================
print(f"🎥 OPENING VIDEO STREAM (SOURCE: {CAMERA_SOURCE})...")
cap = cv2.VideoCapture(CAMERA_SOURCE)

# Fallback to camera 0 if virtual camera 1 is not currently streaming
if not cap.isOpened() and CAMERA_SOURCE == 1:
    print("⚠️ Camera 1 (OBS Virtual Camera) not detected. Falling back to Camera 0 (Webcam)...")
    CAMERA_SOURCE = 0
    cap = cv2.VideoCapture(CAMERA_SOURCE)

if not cap.isOpened():
    print(f"❌ Failed to open video source {CAMERA_SOURCE}.")
    print("👉 If using VDO.Ninja with OBS, click 'Start Virtual Camera' in OBS Studio first.")
    print("👉 Or run: python dustuai.py --cam 0")
    sys.exit(1)

print("🚀 DUSTU_OS VISION SYSTEM READY. Press 'q' in video window to exit.")

# Stats counters for demo HUD
stats_counts = {"DRY": 0, "WET": 0, "EWASTE": 0}

# ==============================================================================
# --- 5. REAL-TIME PERCEPTION & CONTROL LOOP ---
# ==============================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Video frame read failed. Reconnecting...")
        time.sleep(0.5)
        continue

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
            raw_name = model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            track_id = int(box.id[0]) if box.id is not None else -1
            box_area = (x2 - x1) * (y2 - y1)
            area_ratio = box_area / screen_area

            # Proximity collision safety halt
            if area_ratio > 0.60:
                too_close = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

            if conf > 0.45:
                # Classify against our multi-category matrix
                cat_info = classify_object(class_id, raw_name)

                if cat_info is not None:
                    color = cat_info["color"]
                    display_label = f"[{cat_info['tag']}] {cat_info['name']} {conf:.2f}"

                    # Automated Sorting Trigger
                    if cat_info["is_trash"] and area_ratio > TRASH_MIN_AREA:
                        if current_mode in ["WAITING_TRASH", "FOLLOW", "SORT"]:
                            client.publish(TOPIC_CONTROL, "S") 
                            
                            if (current_time - last_mqtt_time > SORT_COOLDOWN):
                                sort_cmd = cat_info["cmd"]
                                cat_name = cat_info["cat"]
                                item_label = cat_info["name"]
                                
                                print(f"♻️ [{cat_name}] {item_label} DETECTED -> DISPATCHING '{sort_cmd}'")
                                client.publish(TOPIC_CONTROL, sort_cmd)
                                client.publish(TOPIC_TELEMETRY, f"{cat_name}: {item_label}")
                                
                                stats_counts[cat_name] = stats_counts.get(cat_name, 0) + 1
                                last_mqtt_time = current_time

                                if current_mode == "WAITING_TRASH":
                                    current_mode = "RETURNING"
                                    return_start_time = current_time + 4
                    
                    elif cat_info["cat"] == "HAZARD":
                        # Hazard alert stops motors immediately
                        client.publish(TOPIC_CONTROL, "S")
                        client.publish(TOPIC_TELEMETRY, f"HAZARD: {cat_info['name']}")

                elif class_id == CLASS_PERSON:
                    color = (0, 255, 0)
                    display_label = f"PERSON {conf:.2f}"

                    # Human tracking and lock
                    if (current_mode in ["FOLLOW", "SUMMON"]):
                        person_center_x = (x1 + x2) // 2
                        if needs_target_lock and abs(person_center_x - frame_center_x) < 100:
                            target_person_id = track_id
                            needs_target_lock = False
                            
                        if track_id == target_person_id:
                            active_target_box = (x1, y1, x2, y2)
                            display_label = f"LOCKED_PILOT [{track_id}]"
                            
                            if current_mode == "SUMMON" and area_ratio > 0.30:
                                current_mode = "WAITING_TRASH"
                                client.publish(TOPIC_CONTROL, "S")
                else:
                    color = (180, 180, 180)
                    display_label = f"{raw_name} {conf:.2f}"

                # Render Bounding Box and High-Tech Tag
                if not too_close:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (lw, lh), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
                    cv2.putText(frame, display_label, (x1 + 3, y1 - 4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # ==============================================================================
    # --- 6. AUTONOMOUS MOVEMENT ENGINE ---
    # ==============================================================================
    if too_close and current_mode != "RETURNING":
        client.publish(TOPIC_CONTROL, "S")
        cv2.putText(frame, "OBSTACLE HALT", (frame_w//2 - 140, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    elif (current_mode in ["FOLLOW", "SUMMON"]) and active_target_box is not None:
        px1, py1, px2, py2 = active_target_box
        person_center_x = (px1 + px2) // 2
        
        if current_time - last_mqtt_time > HEARTBEAT_RATE:
            if person_center_x < (frame_center_x - 60): 
                client.publish(TOPIC_CONTROL, "L")
            elif person_center_x > (frame_center_x + 60): 
                client.publish(TOPIC_CONTROL, "R")
            else: 
                client.publish(TOPIC_CONTROL, "F")
            last_mqtt_time = current_time

    elif current_mode == "RETURNING":
        if current_time > return_start_time: 
            time_spent = current_time - return_start_time
            if time_spent < 0.5: client.publish(TOPIC_CONTROL, "L")
            elif time_spent < 1.0: client.publish(TOPIC_CONTROL, "R")
            else:
                client.publish(TOPIC_CONTROL, "S")
                current_mode = "IDLE"

    # ==============================================================================
    # --- 7. HUD OVERLAY (HEADS-UP DISPLAY) ---
    # ==============================================================================
    # Top status bar
    cv2.rectangle(frame, (0, 0), (frame_w, 38), (15, 15, 20), -1)
    hud_mode_color = (0, 255, 0) if current_mode in ["FOLLOW", "SUMMON", "SORT"] else (0, 200, 255)
    cv2.putText(frame, f"MODE: {current_mode}", (12, 24), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_mode_color, 2)

    stats_str = f"SORTED -> DRY:{stats_counts['DRY']} | WET:{stats_counts['WET']} | E-WASTE:{stats_counts['EWASTE']}"
    cv2.putText(frame, stats_str, (frame_w - 380, 24), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("DUSTU_OS VISION CORTEX", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        current_mode = "SORT"
    elif key == ord('f'):
        current_mode = "FOLLOW"
        needs_target_lock = True
    elif key == ord('h') or key == 32: # spacebar or h
        current_mode = "IDLE"
        client.publish(TOPIC_CONTROL, "S")

# Graceful cleanup
client.publish(TOPIC_CONTROL, "S")
cap.release()
cv2.destroyAllWindows()
print("🛑 DUSTU_OS VISION CORTEX SAFELY SHUT DOWN.")
