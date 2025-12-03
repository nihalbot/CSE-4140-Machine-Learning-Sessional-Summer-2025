from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt") 


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break


    frame = cv2.resize(frame, (640, 480))

    results = model(frame) 
    annotated_frame = results[0].plot() 

    cv2.imshow("YOLO Object Detection", annotated_frame)

  
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
