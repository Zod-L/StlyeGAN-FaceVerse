# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from torchvision.utils import save_image
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures import Meshes
from uv_mesh.mesh import load_obj_mesh
#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, joint_train, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, lap_weight=2, smooth_weight=1, l2_weight=0.05):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.joint_train = joint_train
        self.weights = [1, lap_weight]
        self.smooth_weight = smooth_weight
        self.name = ["im", "lap"]
        self.l2_weight = l2_weight



        _, fa, uva, _ = load_obj_mesh('uv_mesh/Remesh_temp.obj', with_texture=True)
        uv_coords = uva
        uv_coords[:,1] = 1 - uva[:,1]
        self.grid = torch.from_numpy(uv_coords[np.newaxis,:,:].astype('float32')).unsqueeze(2)
        self.grid = self.grid[:,:,:,:2] * 2 - 1
        self.face = torch.from_numpy(fa[np.newaxis,:,:])

    def run_G(self, z, coarse_img, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws, coarse_img)
        return img, ws

    def run_D(self, img, coarse_img, c, name, sync):
        if self.augment_pipe is not None:
            temp = torch.cat((img, coarse_img), dim=1)
            temp = self.augment_pipe(temp)
            img = temp[:,:,:512,:]
        with misc.ddp_sync(self.D, sync):
            logits = self.D(torch.cat((img, coarse_img), 1) , c, name)
        return logits

    def run_deep3d(self):
        raise NotImplementedError

    def accumulate_gradients(self, phase, real_img, coarse_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            if (self.joint_train):
                with torch.autograd.profiler.record_function('deep3d_forward'):
                    coarse_img = self.run_deep3d(coarse_img)
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, coarse_img, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, coarse_img, gen_c, "fake.png", sync=False)
                loss_Gmain = 0
                for i, logit in enumerate(gen_logits):
                    training_stats.report('Loss/scores/fake_' + self.name[i], logit)
                    training_stats.report('Loss/signs/fake_' + self.name[i], logit.sign())
                    loss_Gmain += self.weights[i] * torch.nn.functional.softplus(-logit) # -log(sigmoid(gen_logits))

                training_stats.report('Loss/G/loss', loss_Gmain)

                if self.smooth_weight != 0:
                    vert = torch.nn.functional.grid_sample(gen_img, grid=self.grid.repeat(gen_img.shape[0], 1, 1, 1).to(gen_img.device), mode='bilinear', align_corners=False).squeeze(3).permute((0,2,1))
                    mesh = Meshes(vert, self.face.repeat(gen_img.shape[0], 1, 1).to(gen_img.device))
                    loss_smooth = mesh_laplacian_smoothing(mesh, "cot") * self.smooth_weight
                    training_stats.report('Loss/G/smooth', loss_smooth)

                if self.l2_weight != 0:
                    l2_loss = torch.nn.functional.mse_loss(coarse_img, gen_img) * self.l2_weight
                    training_stats.report('Loss/G/l2', l2_loss)



            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain.mean() + (loss_smooth.mean() if self.smooth_weight != 0 else 0) + (l2_loss.mean() if self.l2_weight != 0 else 0)).mul(gain).backward()
                #loss_Gmain.mean().mul(gain).backward()
            


        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], coarse_img[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight

                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()
            

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, coarse_img, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, coarse_img, gen_c, "fake.png", sync=False) # Gets synced by loss_Dreal.
                loss_Dgen = 0
                for i, logit in enumerate(gen_logits):
                    training_stats.report('Loss/scores/fake_' + self.name[i], logit)
                    training_stats.report('Loss/signs/fake_' + self.name[i], logit.sign())
                    loss_Dgen += self.weights[i] * torch.nn.functional.softplus(logit) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, coarse_img, real_c, "real.png", sync=sync)
                loss_Dreal = 0
                for i, logit in enumerate(real_logits):
                    training_stats.report('Loss/scores/real_' + self.name[i], logit)
                    training_stats.report('Loss/signs/real_' + self.name[i], logit.sign())
                
                    if do_Dmain:
                        loss_Dreal += self.weights[i] * torch.nn.functional.softplus(-logit) # -log(sigmoid(real_logits))
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[sum(real_logits).sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (sum(real_logits) * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
