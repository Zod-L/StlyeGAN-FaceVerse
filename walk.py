import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
from torchvision.utils import save_image
import imageio
import legacy
import pyvista as pv


from render.Renderer_DECA import SRenderY, set_rasterizer
import render.Renderer_DECA_util as util
from render.mesh import load_obj_mesh, save_obj_mesh
from uv_mesh.Remesh_fromUVloc import get_vertex


#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--indir', help='Input image UV dir', type=str, required=True, metavar='DIR')
@click.option('--gtdir', help='GT image UV dir', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    noise_mode: str,
    outdir: str,
    indir: str,
    class_idx: Optional[int],
    truncation_psi,
    gtdir
):

    device = torch.device('cuda', 1)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)


    template = "uv_mesh/Remesh_temp_simp.obj"
    set_rasterizer()  
    render = SRenderY(512, obj_filename=template, uv_size=512, rasterizer_type='pytorch3d').to(device)
    pinCam = torch.FloatTensor([0.8, 0.0, 0.0]).unsqueeze(0).repeat(1,1).to(device)


    # Synthesize the result of a W projection.



    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    fnames = [fname for fname in os.listdir(indir) if fname.endswith(".npy")]
    for fname in tqdm(fnames):
        cond_im = np.load(os.path.join(indir, fname))
        cond_im = torch.from_numpy(cond_im).to(device).permute(2, 0, 1).unsqueeze(0)
        z1 = torch.randn([1, G.z_dim], device=device)
        z2 = torch.randn([1, G.z_dim], device=device)
        video = imageio.get_writer(f'{outdir}/{fname.split(".")[0]}.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')


        uv = np.load(os.path.join(gtdir, fname))
        uv = torch.tensor(uv).to(device).permute(2, 0, 1).unsqueeze(0)
        verts, face = get_vertex(uv)
        trans_verts = util.batch_orth_proj(verts, pinCam)
        trans_verts[:,:,1:] = -verts[:,:,1:]
        img = render(verts, trans_verts, (uv+1.5) * 255/3, h=512, w=512, background=None)['images']
        gt_img = img.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        for i in range(0, 101):
            i /= 100
            z = i * z1 + (1-i) * z2
            uv = G(z=z, cond_im=cond_im, c=label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            verts, face = get_vertex(uv)
            trans_verts = util.batch_orth_proj(verts, pinCam)
            trans_verts[:,:,1:] = -verts[:,:,1:]
            img = render(verts, trans_verts, (uv+1.5) * 255/3, h=512, w=512, background=None)['images']
            img = img.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            #img = render(verts, face)
            video.append_data(np.hstack((img, gt_img)))
        video.close()



def render(vert, face):
    vert = vert.squeeze().cpu().numpy()
    mesh = pv.PolyData(vert, face)
    pl = pv.Plotter(off_screen=True) #create a plot tool
    pl.add_mesh(mesh)
    pl.camera_position = 'xy'
    pl.camera.roll += 0
    pl.camera.azimuth = 15
    pl.camera.elevation = 15

    #pl.camera.position = (1, 1, 0.0)
    #pl.camera.focal_point = (1, 1, 1)
    #pl.camera.up = (0.0, 1.0, 0.0)
    #pl.camera.zoom(1.0)

    pl.enable_eye_dome_lighting() #change render style

    image = pl.screenshot(None, return_img=True)
    return image

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
