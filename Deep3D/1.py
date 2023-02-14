import torch
from scipy.io import loadmat, savemat
from uv_mesh.mesh import load_obj_mesh, save_obj_mesh_with_rgb
import torch.nn.functional as F
import numpy as np
import torch
import cv2
import os
import shutil



# # print(bfm_model["eyebrow_mask"].dtype)
# real_shape = bfm_model["meanshape"].reshape(-1, 3)
# real_tex = bfm_model["meantex"].reshape(-1, 3)
# toon_shape = bfm_model["mean_cartoon_shape"].reshape(-1, 3) 
# toon_tex = bfm_model["mean_cartoon_texture"].reshape(-1, 3) 
# fuse_shape = (real_shape + toon_shape) / 2
# fuse_tex = (real_tex + toon_tex) / 2
# tri = bfm_model["tri"] - 1
# save_obj_mesh_with_rgb("real.obj", real_shape, tri, real_tex)
# save_obj_mesh_with_rgb("toon.obj", toon_shape, tri, toon_tex)
# save_obj_mesh_with_rgb("fuse.obj", fuse_shape, tri, fuse_tex)




# _,_,real_mask = load_obj_mesh("real_mask.obj")
# real_idx = np.all(real_mask == [1,0,0], axis=-1)



# shape = (bfm_model["meanshape"] + bfm_model["mean_cartoon_shape"]) / 2
# tex = (bfm_model["meantex"] + bfm_model["mean_cartoon_texture"]) / 2
# tex = tex.reshape(-1, 3)
# tex[real_idx] = bfm_model["meantex"].reshape(-1,3)[real_idx]
# tri = bfm_model["tri"] 
# save_obj_mesh_with_rgb("mean_rmtoonme.obj", shape.reshape(-1, 3), tri-1, tex)


im_path = "/home/liyi/data/faceverse_data/train/im"
lm_path = "/home/liyi/data/faceverse_data/train/im/landmarks"
mask_path = "/home/liyi/data/faceverse_data/train/im/mask"
fnames = [fname for fname in os.listdir(mask_path)]


with open("Deep3D/datalist/real/masks.txt", "w") as fh:
    for fname in fnames:
        fh.write(os.path.join(mask_path, fname) + "\n")

with open("Deep3D/datalist/real/images.txt", "w") as fh:
    for fname in fnames:
        fh.write(os.path.join(im_path, fname) + "\n")

with open("Deep3D/datalist/real/landmarks.txt", "w") as fh:
    for fname in fnames:
        fh.write(os.path.join(lm_path, fname.split(".")[-2] + ".txt") + "\n")

