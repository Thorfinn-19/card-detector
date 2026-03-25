from ultralytics import YOLO

model = YOLO("C:/Gepi latas/card-detector/yolov8nbest.pt")
model.predict(source="C:/Users/ukogn/Downloads/istockphoto-1131822944-612x612.jpg", save=True)