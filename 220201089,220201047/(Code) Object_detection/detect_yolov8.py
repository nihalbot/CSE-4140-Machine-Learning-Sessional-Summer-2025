import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Unable to access the camera.")
    exit()

print("🟢 YOLOv8 is running... Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("👋 ESC pressed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
