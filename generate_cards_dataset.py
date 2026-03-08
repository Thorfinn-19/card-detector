import cv2
from pathlib import Path
import random
import numpy as np

CARDS_DIR = Path("assets/cards")
TEXTURE_DIR = Path("assets/texture")
DATASET_DIR = Path("datasets/yolo_cards")

card_paths = list(CARDS_DIR.glob("*.png"))
bg_paths = list(TEXTURE_DIR.glob("*.jpg"))

# Osztályrendszer
class_names = sorted([p.stem for p in card_paths])
class_to_id = {name: i for i, name in enumerate(class_names)}

# cards.yaml
yaml_path = Path("cards.yaml")
with open(yaml_path, "w") as f:
    f.write("path: datasets/yolo_cards\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n\n")
    f.write("names:\n")
    for i, name in enumerate(class_names):
        f.write(f"  {i}: {name}\n")

# Képek betöltése
cards = {p.stem: cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in card_paths}
backgrounds = [cv2.imread(str(p)) for p in bg_paths]

IMG_SIZE = 640

def rotate(card, angle):
    h, w = card.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += new_w / 2 - w / 2
    M[1, 2] += new_h / 2 - h / 2

    return cv2.warpAffine(card, M, (new_w, new_h), borderValue=(0, 0, 0, 0))

def blend(bg, card, x, y):
    h, w = card.shape[:2]
    roi = bg[y:y+h, x:x+w].astype(np.float32)
    rgb = card[:, :, :3].astype(np.float32)
    alpha = (card[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    bg[y:y+h, x:x+w] = np.clip(alpha * rgb + (1 - alpha) * roi, 0, 255).astype(np.uint8)

def alpha_bbox(card, x, y):
    alpha = card[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return None
    x1 = xs.min() + x
    y1 = ys.min() + y
    x2 = xs.max() + 1 + x
    y2 = ys.max() + 1 + y
    return x1, y1, x2, y2

def yolo_label(class_id, box, img_size=640):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2
    cy = y1 + bh / 2
    return f"{class_id} {cx/img_size:.6f} {cy/img_size:.6f} {bw/img_size:.6f} {bh/img_size:.6f}"

def make_dirs():
    for split in ["train", "val", "test"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def generate_one_image():
    bg = random.choice(backgrounds).copy()
    chosen_cards = random.sample(class_names, 5)

    placed_centers = []
    labels = []

    for name in chosen_cards:
        card = cards[name]

        scale = random.uniform(0.30, 0.45)
        h, w = card.shape[:2]
        new_h = int(scale * IMG_SIZE)
        sc = new_h / h
        card = cv2.resize(card, (int(w * sc), new_h))

        card = rotate(card, random.uniform(-35, 35))
        ch, cw = card.shape[:2]

        for _ in range(50):
            x = random.randint(0, IMG_SIZE - cw)
            y = random.randint(0, IMG_SIZE - ch)
            cx, cy = x + cw / 2, y + ch / 2

            if any((cx - px) ** 2 + (cy - py) ** 2 < 40 ** 2 for px, py in placed_centers):
                continue

            placed_centers.append((cx, cy))
            blend(bg, card, x, y)

            box = alpha_bbox(card, x, y)
            if box is not None:
                labels.append(yolo_label(class_to_id[name], box))
            break

    return bg, labels

def save_split(split, count, start_idx=0):
    for i in range(count):
        img, labels = generate_one_image()
        idx = start_idx + i

        img_path = DATASET_DIR / "images" / split / f"img_{idx:05d}.jpg"
        txt_path = DATASET_DIR / "labels" / split / f"img_{idx:05d}.txt"

        cv2.imwrite(str(img_path), img)

        with open(txt_path, "w") as f:
            for line in labels:
                f.write(line + "\n")

make_dirs()

save_split("train", 4000, 0)
save_split("val", 500, 4000)
save_split("test", 500, 4500)