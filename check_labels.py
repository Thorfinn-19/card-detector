import cv2
from pathlib import Path

IMG_PATH = Path("datasets/yolo_cards/images/train/img_00223.jpg")
LABEL_PATH = Path("datasets/yolo_cards/labels/train/img_00223.txt")

img = cv2.imread(str(IMG_PATH))
h, w = img.shape[:2]

with open(LABEL_PATH, "r") as f:
    lines = f.readlines()

for line in lines:
    class_id, cx, cy, bw, bh = line.strip().split()
    class_id = int(class_id)
    cx, cy, bw, bh = map(float, [cx, cy, bw, bh])

    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, str(class_id), (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("Check labels", img)
cv2.waitKey(0)
cv2.destroyAllWindows()