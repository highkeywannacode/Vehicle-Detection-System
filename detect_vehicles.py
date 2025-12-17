import cv2
from ultralytics import YOLO

# 1. LOAD YOUR MODEL
model = YOLO('models/best.pt')

# 2. OPEN VIDEO
cap = cv2.VideoCapture("test_video.mp4")

# 3. SETUP SAVING
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# using 'mp4v' codec usually works best for Windows
out = cv2.VideoWriter('result_debug.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print("Processing... Watch the terminal!")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break 
    
    frame_count += 1

    # 4. RUN DETECTION (Reduced confidence to 0.1 to force boxes)
    results = model(frame, conf=0.1) 

    detections = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            detections += 1
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw Box (Thicker line = 3)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Label
            cls = int(box.cls[0])
            conf_score = float(box.conf[0])
            label = f"{model.names[cls]} {conf_score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Debug Print
    print(f"Frame {frame_count}: Found {detections} vehicles")

    # 5. SHOW & SAVE
    cv2.imshow("Debug View", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()