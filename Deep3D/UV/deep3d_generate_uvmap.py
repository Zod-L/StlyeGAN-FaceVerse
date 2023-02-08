''' 
Generate 2d uv maps representing different attributes(colors, depth, image position, etc)
: render attributes to uv space.
'''
import os, sys
import numpy as np
import scipy.io as sio
from skimage import io
import skimage.transform
from time import time
import matplotlib.pyplot as plt
from uv_mesh.mesh import load_obj_mesh

import cv2

sys.path.append('..')
import face3d
from face3d import mesh
from face3d.morphable_model import MorphabelModel

def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords


def uv_expand(shape, color):    
    verts, faces, uvs, face_uvs = load_obj_mesh('./UV/texTemp.obj', with_texture=True) # deep3d uv unwrap
    uv_coords = uvs
    triangles = face_uvs+1

    size = 512
    uv_h = uv_w = size
    image_h = image_w = size
    uv_coords = process_uv(uv_coords, uv_h, uv_w)

    _,fc,c = load_obj_mesh('./UV/temp1.obj')
    ## 去掉标记面片
    index = c[:,1] <0.4

    buf = []
    for i in range(fc.shape[0]):
        flag = False
        for j in fc[i,:]:
            if index[j]:
                flag = True
                break
        
        if not flag:
            buf.append(i)

    #fc = fc[buf,:] 
    uv_tex_map = mesh.render.render_colors(uv_coords, fc, color, uv_h, uv_w, c=3)
    uv_loc_map = mesh.render.render_colors(uv_coords, fc, shape, uv_h, uv_w, c=3)
    
    
    return uv_loc_map, uv_tex_map

