from ultralytics import YOLO

model = YOLO("C:/Downloads/best.pt")
model.predict(source="C:/card-detector/datasets/yolo_cards/images/test/img_00922.jpg", save=True)