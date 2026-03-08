from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="cards.yaml", 
    imgsz=512,
    epochs=50,
    batch=-1,
    device=0,
    workers=2,
    project="runs",
    name="card_detector"
)
from google.colab import files
files.download('/runs/card_detector/weights/best.pt')