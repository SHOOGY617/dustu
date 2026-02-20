import cv2
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import time

# --- 1. CONFIGURATION ---
# REPLACE THIS with the IP shown in your "IP Webcam" app
# Example: "http://192.168.1.5:8080/video"
CAMERA_URL = 1

# MQTT Settings (Must match your NodeMCU code)
MQTT_BROKER = "broker.emqx.io"
TOPIC = "dustu/control"

# --- 2. DEFINE TRASH ---
# YOLOv8 Class IDs (COCO Dataset)
# 39: Bottle, 41: Cup, 46: Banana, 47: Apple, 49: Orange, 50: Broccoli
WET_ITEMS = [46, 47, 49, 50, 51]  # Fruits/Veg
DRY_ITEMS = [39, 41, 67]          # Bottles, Cups, Cell Phones

# --- 3. INITIALIZE SYSTEM ---
print("🧠 LOADING AI BRAIN... (This may take a moment)")
model = YOLO("yolov8n.pt")  # Load the Nano model (Fastest)

print("🌐 CONNECTING TO CLOUD...")
client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)
client.loop_start()
print("✅ CONNECTED!")

# Start Video Stream
cap = cv2.VideoCapture(CAMERA_URL)

# Variables for "Cooldown" (To stop spamming)
last_cmd_time = 0
COOLDOWN = 10  # Seconds to wait between sorting

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Cannot see video. Check IP Webcam App.")
        break

    # --- 4. THE AI VISION ---
    # Run YOLOv8 on the frame
    results = model(frame, stream=True, verbose=False)

    current_time = time.time()
    
    # Process what the AI sees
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get Object Details
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[class_id]
            
            # Draw Box on Screen
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # --- 5. THE DECISION LOGIC ---
            # Only act if confidence is high (> 50%) and Cooldown is over
            if conf > 0.5:
                # Color the box based on type
                color = (255, 0, 0) # Blue default
                
                if (current_time - last_cmd_time > COOLDOWN):
                    
                    if class_id in DRY_ITEMS:
                        print(f"🧃 DETECTED: {name} --> SORTING DRY")
                        client.publish(TOPIC, "DRY")  # SEND 'D' TO ROBOT
                        last_cmd_time = current_time
                        color = (0, 255, 0) # Green for Dry
                        
                    elif class_id in WET_ITEMS:
                        print(f"🍎 DETECTED: {name} --> SORTING WET")
                        client.publish(TOPIC, "WET")  # SEND 'W' TO ROBOT
                        last_cmd_time = current_time
                        color = (0, 0, 255) # Red for Wet
                
                # Draw the box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show the "Terminator View" on Laptop
    cv2.imshow("DUSTU AI VISION", frame)

    # Press 'Q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()