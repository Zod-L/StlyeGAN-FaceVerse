import numpy as np
from uv_mesh.mesh import load_obj_mesh
import cv2
import os
import shutil


    


x = np.load("/home/liyi/data/faceverse_data/train/uv/fine/loc/00012.npy")
y = np.load("/home/liyi/data/faceverse_data/train/uv/coarse/loc/00012.npy")
print(x.min(), x.max())
print(y.min(), y.max())
print((x-y).min(), (x-y).max())

