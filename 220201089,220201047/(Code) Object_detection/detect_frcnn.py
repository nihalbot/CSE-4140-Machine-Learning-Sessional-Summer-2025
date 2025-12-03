import torch
import torchvision
from torchvision.transforms import functional as F
import cv2


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
faster_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
faster_model.to(device)
faster_model.eval()


try:
    from ultralytics import YOLO
    yolo_model = YOLO('yolov8n.pt') 
except ImportError:
    print("Install ultralytics: pip install ultralytics")
    yolo_model = None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(rgb_frame).to(device)

    with torch.no_grad():
        predictions = faster_model([img_tensor])[0]

    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"FRCNN: {COCO_INSTANCE_CATEGORY_NAMES[label.item()]} {score:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)


    if yolo_model:
        results = yolo_model(frame, verbose=False)[0]
        for r in results.boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
            conf = r.conf[0].item()
            cls = int(r.cls[0].item())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f"YOLO: {results.names[cls]} {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

    cv2.imshow('Real-Time Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
