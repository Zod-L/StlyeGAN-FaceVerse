import os
from PIL import Image
import numpy as np
import cv2
import torch



dir = "/home/liyi/data/faceverse_data/train/uv/fine/tex"


for fname in os.listdir(dir):
    if not fname.endswith(".npy"):
        continue

    im = np.load(os.path.join(dir, fname)).astype(np.float32)
    print(im.min(), im.max())