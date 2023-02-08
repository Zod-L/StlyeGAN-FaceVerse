import torch
from scipy.io import loadmat, savemat
from uv_mesh.mesh import load_obj_mesh, save_obj_mesh_with_rgb
import torch.nn.functional as F
import numpy as np
import torch
import cv2
import os
import shutil

bfm_model = loadmat("Deep3D/BFM/fuse_model_front_improve.mat")
for k,v in bfm_model.items():
    try:
        print(f"{k} : {v.shape}")
    except:
        pass
meanshape = bfm_model["meanshape"].reshape(-1, 3)
translate = np.array([(meanshape[:, 0].max()+meanshape[:, 0].min())/2, (meanshape[:, 1].max()+meanshape[:, 1].min())/2, (meanshape[:, 2].max()+meanshape[:, 2].min())/2])
print((meanshape - translate).min())
print((meanshape - translate).max())

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




