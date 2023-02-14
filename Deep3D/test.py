"""This script is the test script for Deep3DFaceRecon_pytorch
"""

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch 
from data.flist_dataset import default_flist_reader
from scipy.io import loadmat, savemat
import torchvision.transforms as transforms
import sys
sys.path.append('/path/to/search')

def get_data_path(opt):
    with open(os.path.join(opt.datalist, "images.txt")) as fh:
        im_path = [os.path.join(opt.root, i.rstrip("\n")) for i in fh]
    with open(os.path.join(opt.datalist, "landmarks.txt")) as fh:
        #lm_path = [os.path.join(opt.root, i.rstrip("\n").replace("landmarks", "detections")) for i in fh]
        lm_path = [os.path.join(opt.root, i.rstrip("\n")) for i in fh]
    with open(os.path.join(opt.datalist, "masks.txt")) as fh:
        mask_path = [os.path.join(opt.root, i.rstrip("\n")) for i in fh]


    return im_path, lm_path, mask_path, im_path[0].split(os.path.sep)[-2]

# def read_data(im_path, lm_path, mask_path, lm3d_std, to_tensor=True):
#     # to RGB 
#     im = Image.open(im_path).convert('RGB')
#     W,H = im.size
#     mask = Image.open(mask_path).convert('RGB')
#     lm = np.loadtxt(lm_path).astype(np.float32)
#     lm[:, -1] = H - 1 - lm[:, -1]
#     _, im, lm, mask = align_img(im, lm, lm3d_std, mask)
    
#     if to_tensor:
#         op = transforms.ToTensor()
#         im = op(im).unsqueeze(0)
#         lm = op(np.array(lm).astype(np.float32)).unsqueeze(0)
#         lm[..., 1] = H - 1 - lm[..., 1]
#         mask = op(mask).unsqueeze(0)
#     return im, lm, mask

def read_data(im_path, lm_path, mask_path, lm3d_std, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')
    W,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    #lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, mask = align_img(im, lm, lm3d_std, mask)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        mask = torch.tensor(np.array(mask)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm, mask



def main(rank, opt):
    device = torch.device('cuda', rank)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)

    im_path, lm_path, mask_path, name = get_data_path(opt)
    lm3d_std = load_lm3d(opt.bfm_folder) 

    for i in range(len(im_path)):
        print(i, im_path[i])
        img_name = im_path[i].split(os.path.sep)[-1].replace('.png','').replace('.jpg','')
        if not os.path.isfile(lm_path[i]):
            print("%s is not found !!!"%lm_path[i])
            continue
        im_tensor, lm_tensor, msk_tensor = read_data(im_path[i], lm_path[i], mask_path[i], lm3d_std)

        data = {
            'imgs': im_tensor,
            'lms': lm_tensor,
            'msks': msk_tensor 
        }
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        visualizer.display_current_results(visuals, 0, opt.epoch, dataset=name, 
            save_results=True, count=i, name=img_name, add_image=False)

        model.save_mesh(os.path.join(visualizer.img_dir, name, 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.obj')) # save reconstruction meshes
        if not os.path.exists(os.path.join(visualizer.img_dir, name, 'epoch_%s_%06d'%(opt.epoch, 0), "tex")):
            os.mkdir(os.path.join(visualizer.img_dir, name, 'epoch_%s_%06d'%(opt.epoch, 0), "tex"))
        if not os.path.exists(os.path.join(visualizer.img_dir, name, 'epoch_%s_%06d'%(opt.epoch, 0), "shape")):
            os.mkdir(os.path.join(visualizer.img_dir, name, 'epoch_%s_%06d'%(opt.epoch, 0), "shape"))
        model.save_uv(os.path.join(visualizer.img_dir, name, 'epoch_%s_%06d'%(opt.epoch, 0), "tex", img_name+'.npy'), \
            os.path.join(visualizer.img_dir, name, 'epoch_%s_%06d'%(opt.epoch, 0), "shape", img_name+'.npy'))
        #model.save_coeff(os.path.join(visualizer.img_dir, name, 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.mat')) # save predicted coefficients

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.continue_train = False
    main(0, opt)
    
