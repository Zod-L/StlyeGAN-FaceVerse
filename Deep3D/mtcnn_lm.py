from facenet_pytorch import MTCNN
from PIL import Image
import os
from tqdm import tqdm
import torch

img_path = "Deep3D/cap"
detect_path = "Deep3D/cap/detections"

fnames = [fname for fname in os.listdir(img_path)]
mtcnn = MTCNN(keep_all=True)

if not os.path.exists(detect_path):
    os.mkdir(detect_path)
for fname in tqdm(fnames):
    img = Image.open(os.path.join(img_path, fname))
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
    with open(os.path.join(detect_path, fname.split(".")[0] + ".txt"), "w") as fh:
        for i in range(5):
            if i < 4:
                fh.write(f"{landmarks[0, i, 0]} {landmarks[0, i, 1]}\n")
            else:
                fh.write(f"{landmarks[0, i, 0]} {landmarks[0, i, 1]}")
