"""C. P. Burgess et al., "MONet: Unsupervised Scene Decomposition and Representation," pp. 1–22, 2019."""
from itertools import chain

import torch
from torch import nn, optim

from .base_model import BaseModel
from . import networks
from .inverse_graphics import Mask2Cube, Cube2Mask
import os.path as osp
from imageio import imwrite
import numpy as np


class MONetModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(batch_size=64, lr=1e-4, display_ncols=7, niter_decay=0,
                            dataset_mode='clevr', niter=int(64e6 // 7e4))
        parser.add_argument('--num_slots', metavar='K', type=int, default=5, help='Number of supported slots')
        parser.add_argument('--frames', type=int, default=1, help='Number of frames to stack together as input')
        parser.add_argument('--z_dim', type=int, default=16, help='Dimension of individual z latent per slot')
        if is_train:
            parser.add_argument('--beta', type=float, default=0.5, help='weight for the encoder KLD')
            parser.add_argument('--gamma', type=float, default=0.5, help='weight for the mask KLD')
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        if opt.depth:
            opt.input_nc = 4
        if opt.optical_flow:
            opt.input_nc = 5

        self.loss_names = ['E', 'D', 'mask']

        if opt.refine_motion:
            self.loss_names.append('bce')

        if opt.physics_loss:
            self.loss_names.append('physics')
            self.loss_names.append('prim_next')

        self.opt = opt
        self.visual_names = [['{}m{}'.format(j, i) for i in range(opt.num_slots)] + \
                            ['{}x{}'.format(j, i) for i in range(opt.num_slots)] + \
                            ['{}xm{}'.format(j, i) for i in range(opt.num_slots)] + \
                            ['{}x'.format(j), '{}x_tilde'.format(j)] for j in range(opt.frames)]

        self.visual_names = sum(self.visual_names, [])
        print(self.visual_names)
        self.model_names = ['Attn', 'CVAE']

        if self.opt.monet_baseline:
            self.netAttn = networks.init_net(networks.AttentionOld(opt.input_nc * opt.frames, opt.frames, ngf=2*opt.z_dim), gpu_ids=self.gpu_ids)
            self.netCVAE = networks.init_net(networks.ComponentVAEOld(opt.input_nc, opt.z_dim, frames=opt.frames), gpu_ids=self.gpu_ids)
        else:
            self.netAttn = networks.init_net(networks.Attention(opt.input_nc * opt.frames, opt.frames, ngf=2*opt.z_dim), gpu_ids=self.gpu_ids)
            self.netCVAE = networks.init_net(networks.ComponentVAE(opt.input_nc, opt.z_dim, frames=opt.frames), gpu_ids=self.gpu_ids)
        # snapshot = torch.load("/data/vision/billf/scratch/yilundu/adept/adept_seg3d/cachedir/smoketest/model_43000")
        if opt.dataset_mode == "block":
            snapshot = torch.load("/data/vision/billf/scratch/yilundu/adept/adept_seg3d/cachedir//block_129/model_10000")
        else:
            # Shapenet objects
            snapshot = torch.load("/data/vision/billf/scratch/yilundu/adept/adept_seg3d/cachedir/seg_mask_again_126/model_35000")
            # Only blocks
            # snapshot = torch.load("/data/vision/billf/scratch/yilundu/adept/adept_seg3d/cachedir/smoketest/model_43000")
        self.mask2prim = Mask2Cube()
        self.prim2mask = Cube2Mask()
        self.mask2prim.load_state_dict(snapshot['mask2cube_state_dict'])
        self.prim2mask.load_state_dict(snapshot['cube2mask_state_dict'])
        self.mask2prim = self.mask2prim.cuda().eval()
        self.prim2mask = self.prim2mask.cuda().eval()
        # The camera these models are trained at is positioned at (5.266, 0, 2.462)
        self.camera_pose = torch.Tensor([5.266, 0, 2.462]).to(self.device)

        self.counter = 0
        self.eps = torch.finfo(torch.float).eps
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        if self.isTrain:  # only defined during training time
            self.criterionKL = nn.KLDivLoss(reduction='batchmean')
            self.optimizer = optim.RMSprop(chain(self.netAttn.parameters(), self.netCVAE.parameters()), lr=opt.lr)
            self.optimizers = [self.optimizer]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        opt = self.opt

        if self.opt.physics_loss:
            self.x = input
            size = self.x.size()
            self.im_size = size
            self.x = self.x.view(-1, *size[2:])
            self.x = self.x.to(self.device)
        else:
            input, seg_im = input
            self.x = input.to(self.device)
            self.seg_im = seg_im.to(self.device)
            self.image_paths = "junk"

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.loss_E = 0
        self.x_tilde = 0
        b = []
        m = []
        m_tilde_logits = []

        # Initial s_k = 1: shape = (N, 1, H, W)
        shape = list(self.x.shape)
        shape[1] = self.opt.frames
        log_s_k = self.x.new_zeros(shape)

        decode_objs = []

        if self.opt.full_robonet or self.opt.eval_intphys or self.opt.eval_cube:
            context = torch.no_grad
        else:
            context = torch.enable_grad

        with context():
            for k in range(self.opt.num_slots):
                # Derive mask from current scope
                if k != self.opt.num_slots - 1:
                    log_alpha_k = self.netAttn(self.x, log_s_k)
                    log_m_k = log_s_k + log_alpha_k
                    # Compute next scope
                    log_s_k += (1. - log_alpha_k.exp()).clamp(min=self.eps).log()
                else:
                    log_m_k = log_s_k


                # Get component and mask reconstruction, as well as the z_k parameters
                m_tilde_k_logits, x_mu_k, x_logvar_k, z_mu_k, z_logvar_k = self.netCVAE(self.x, log_m_k, k == 0)

                if self.opt.dataset_mode == "block":
                    if k != 0:
                        decode_objs.append(m_tilde_k_logits)
                elif k != 0 and k != self.opt.num_slots - 1:
                    decode_objs.append(m_tilde_k_logits)

                # KLD is additive for independent distributions
                self.loss_E += -0.5 * (1 + z_logvar_k - z_mu_k.pow(2) - z_logvar_k.exp()).sum()

                m_k = log_m_k.exp()

                if self.opt.frames > 1:
                    s = x_mu_k.size()
                    x_mu_k_expand = x_mu_k.view(s[0], self.opt.frames, 3, *s[2:])
                    x_k_masked = x_mu_k_expand * m_k[:, :, None, :, :]
                    x_k_masked = x_k_masked.view(*s)
                else:
                    x_k_masked = m_k * x_mu_k


                # Exponents for the decoder loss
                if self.opt.frames > 1:
                    s = log_m_k.size()
                    log_m_k = log_m_k[:, :, None, :, :].repeat(1, 1, 3, 1, 1)
                    log_m_k = log_m_k.view(s[0], self.opt.frames * 3, *s[2:])
                    b_k = log_m_k - 0.5 * x_logvar_k - (self.x - x_mu_k).pow(2) / (2 * x_logvar_k.exp())
                else:
                    b_k = log_m_k - 0.5 * x_logvar_k - (self.x - x_mu_k).pow(2) / (2 * x_logvar_k.exp())
                b.append(b_k.unsqueeze(1))

                # Get outputs for kth step

                # Iteratively reconstruct the output image
                self.x_tilde += x_k_masked

                for i in range(self.opt.frames):
                    setattr(self, '{}m{}'.format(i, k), m_k[:, i:i+1] * 2. - 1.) # shift mask from [0, 1] to [-1, 1]
                    setattr(self, '{}x{}'.format(i, k), x_mu_k[:, 3*i:3*(i+1)])
                    setattr(self, '{}xm{}'.format(i, k), x_k_masked[:, 3*i:3*(i+1)])
                    setattr(self, '{}x_tilde'.format(i), self.x_tilde[:, 3*i:3*(i+1)])
                    setattr(self, '{}x'.format(i), self.x[:, 3*i:3*(i+1)])

                # Accumulate
                m.append(m_k)
                m_tilde_logits.append(m_tilde_k_logits)

            self.decode_objs = torch.cat(decode_objs, dim=1)

        if self.opt.physics_loss:
            masks = self.decode_objs.exp()
            select_mask = masks

            if self.opt.dataset_mode == "block":
                null_tresh = 300
            else:
                null_tresh = 50

            # For now let's use the threshold pixel of 50
            mask_valid = select_mask.sum(dim=[2, 3]) > null_tresh
            s = mask_valid.size()
            sm = select_mask.size()
            select_mask = select_mask.view(sm[0]*sm[1], 1, sm[2], sm[3])

            prim = self.mask2prim(select_mask)
            prim = prim.view(-1, 3, s[1], 7)


            prim_diff_pos = prim[:, 1, :3] - prim[:, 0, :3]
            prim_next = torch.cat([prim[:, 1, :3] + prim_diff_pos, (prim[:, 1, 3:5] + prim[:, 0, 3:5]) / 2., 2 * prim[:, 1, 5:] - prim[:, 0, 5:]], dim=1)
            # prim_next = prim[:, 1].contiguous()

            s = prim_next.size()

            loss_prim_next = torch.pow(prim_next - prim[:, 2], 2).mean()
            prim_next = prim_next.view(-1, 7)

            mask_pred = self.prim2mask(prim_next)
            ms = mask_pred.size()
            mask_pred = mask_pred.view(s[0], s[1], *ms[1:])
            prim_next = prim_next.view(*s)

            # Use the last state to infer the set of valid objects
            mask_valid = mask_valid.view(-1, 3, s[1])[:, 1, :]
            dist = torch.norm(prim_next[:, :, :3] - self.camera_pose[None, None, :], p=2, dim=2)

            dist_idx = torch.argsort(dist, dim=1)
            mask_pred_total = torch.zeros_like(mask_pred[:, 0:1])
            mask_pred_filter = torch.zeros_like(mask_pred)
            select_mask = select_mask.view(-1, 3, sm[1], 1, sm[2], sm[3])
            # Filter through each element of the predict mask based off distance to handle occlusions between objects

            # dist_idx is size
            # 6 x 3
            # mask_valid is also size
            # 6 x 3
            for i in range(dist_idx.size(0)):
                for j in range(dist_idx.size(1)):
                # Could probably try to parallelize this operation
                    select_idx = dist_idx[i, j]
                    valid_mask = (mask_pred_total[i, 0] < 0.4).float()
                    mask_pred_filter[i, select_idx] = mask_pred[i, select_idx] * valid_mask * mask_valid[i, select_idx]
                    mask_pred_total[i, 0] = mask_pred_total[i, 0] + mask_pred[i, select_idx] * mask_valid[i, select_idx] * valid_mask

            # Some code for debugging values
            # init_image = select_mask.detach().cpu().numpy()
            # s = init_image.shape
            # import pdb
            # pdb.set_trace()
            # init_image = init_image.reshape((s[2] * s[0] * s[1], s[3]))
            # imwrite("init_im.png", init_image)

            if self.opt.dataset_mode == "block":
                mask_tresh = 0.4
                mask_tresh_other = 0.4
            else:
                mask_tresh = 0.8
                mask_tresh_other = 0.2

            # select_mask_im = select_mask[:, :, :, 0].detach().cpu().numpy()
            # mask_pred_filter_im = (mask_pred_filter[:, :, 0] > mask_tresh).detach().cpu().numpy()

            # for i in range(select_mask_im.shape[0]):
            #     select_mask_im_i = select_mask_im[i]
            #     mask_pred_filter_im_i = mask_pred_filter_im[i]
            #     joint_im = np.concatenate([select_mask_im_i, mask_pred_filter_im_i[None, :]], axis=0)
            #     joint_im = joint_im.transpose((0, 2, 1, 3))
            #     s = joint_im.shape
            #     joint_im = joint_im.reshape((s[0]*s[1], s[2]*s[3]))
            #     imwrite(osp.join("masks", "joint_{}.png".format(self.counter)), joint_im)
            #     self.counter += 1

            # if self.counter == 50:
            #     assert False

            # Make the masks the same
            mask_label = (mask_pred_filter > mask_tresh).float()
            train_instance = select_mask[:, 2]
            train_label = (train_instance > mask_tresh_other).float()
            # loss_physics = torch.pow(mask_pred_filter - select_mask[:, 2], 2).mean()
            loss_physics = (((-mask_label) * (train_instance + 1e-5).log() - (1 - mask_label) * (1 - train_instance + 1e-5).log())).mean()
            loss_physics = ((-train_label) * (mask_pred_filter + 1e-5).log() - (1 - train_label) * (1 - mask_pred_filter + 1e-5).log()).mean() + loss_physics

            # This is the solution for adept
            # self.loss_physics = loss_physics * 100.0
            # self.loss_prim_next = loss_prim_next * 1.0

            # This is the solution for blocks
            # self.loss_physics = loss_physics * 10.0
            # self.loss_prim_next = loss_prim_next * 10.0

            # Else
            self.loss_physics = loss_physics * 100.0
            self.loss_prim_next = loss_prim_next * 1.0


        self.b = torch.cat(b, dim=1)
        self.m = torch.cat(m, dim=1)
        self.m_tilde_logits = torch.cat(m_tilde_logits, dim=1)

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        n = self.x.shape[0]
        self.loss_E /= n
        self.loss_D = -torch.logsumexp(self.b, dim=1).sum() / n
        self.loss_mask = self.criterionKL(self.m_tilde_logits.log_softmax(dim=1), self.m)
        loss = self.loss_D + self.opt.beta * self.loss_E + self.opt.gamma * self.loss_mask

        if self.opt.physics_loss:
            loss = loss + self.loss_physics + self.loss_prim_next

        if self.opt.refine_motion:
            decode_objs = self.decode_objs
            seg_im = self.seg_im
            valid = (self.seg_im).sum(dim=[1, 2]) > 0
            seg_im, decode_objs = seg_im[valid], decode_objs[valid]

            if seg_im.size(0) > 0:
                # Compute a forward loss
                s = seg_im.size()
                overlap_val = (seg_im[:, None, :] * decode_objs.exp()).sum(dim=[2, 3])
                select_idx = torch.argmax(overlap_val, dim=1)
                select_obj = torch.gather(decode_objs, 1, select_idx[:, None, None, None].repeat(1, 1, s[1], s[2]))
                bce_loss = (-seg_im * (select_obj + 1e-5) + -(1 - seg_im) * torch.log(1 - select_obj.exp() + 1e-5)).mean()
                self.loss_bce = bce_loss
                loss = loss + 10 * bce_loss
        loss.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        self.optimizer.step()        # update gradients for network G
