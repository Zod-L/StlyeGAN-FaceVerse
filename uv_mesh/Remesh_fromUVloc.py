import os, sys
import numpy as np
#from uv_mesh.mesh import load_obj_mesh, save_obj_mesh
from mesh import load_obj_mesh, save_obj_mesh

import matplotlib.pyplot as plt
import pymeshlab

import torch
from torch.nn import functional as F
import cv2
from tqdm import tqdm


def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    uv_coords = uv_coords * (uv_w - 1)
    #uv_coords = scale_uv(uv_coords)
    
    #uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    #uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords


def get_vertex(img):
    _, fa, uva, _ = load_obj_mesh('uv_mesh/Remesh_temp_simp.obj', with_texture=True)

    uv_coords = uva
    uv_coords[:,1] = 1 - uva[:,1]
    grid = torch.from_numpy(uv_coords[np.newaxis,:,:].astype('float32')).unsqueeze(2).to(img.device)
    grid = grid[:,:,:,:2] * 2 - 1

    loc = F.grid_sample(img, grid=grid, mode='bilinear', align_corners=False)
    return loc.squeeze(3).permute((0,2,1)), fa






if __name__ == "__main__":
    #_, fa, uva, _ = load_obj_mesh('uv_mesh/Remesh_temp_simp.obj', with_texture=True)
    _, fa, uva, _ = load_obj_mesh('uv_mesh/Remesh_temp.obj', with_texture=True)

    uv_coords = uva
    uv_coords[:,1] = 1 - uva[:,1]
    grid = torch.from_numpy(uv_coords[np.newaxis,:,:].astype('float32')).unsqueeze(2) # [B, N, 1, 2]
    grid = grid[:,:,:,:2] * 2 - 1

    path = './generate_out/'
    # gt_path = "/home/liyi/data/faceverse_data/train/uv/fine/loc/remesh/"
    op = path + 'remesh/'

    files = [file for file in os.listdir(path) if file.endswith(".npy")]

    if not os.path.exists(op):
        os.mkdir(op)
        

    l2 = 0

    for i in tqdm(files):
        img = np.load(path + i)
        temp = torch.from_numpy(img[np.newaxis,:,:,:].astype('float32')).permute(0,3,1,2)#/280
        loc = F.grid_sample(temp, grid=grid, mode='bilinear', align_corners=False).squeeze()
        loc = loc.numpy().squeeze().T

        file_name = op + f'{i[:-4]}.obj'
        lp_name = op + f'{i[:-4]}_lap.obj'
        # gt, _ = load_obj_mesh(gt_path + f'{i[:-4]}.obj')

        # l2 += np.mean(((gt+1.5) * 255/3 - (loc+1.5) * 255/3) ** 2)
        # print(l2)


        
        save_obj_mesh(file_name, loc, fa)
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file_name)
        ms.apply_coord_laplacian_smoothing(stepsmoothnum=2, boundary=True, cotangentweight=False)
        ms.save_current_mesh(lp_name, save_vertex_normal=False)



