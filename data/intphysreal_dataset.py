import os
import os.path as osp
import numpy as np

import torchvision.transforms.functional as TF
import random

from PIL import Image
import torch.utils.data as data
from scipy.misc import imread
from scipy.misc import imsave
import torch

from scipy.misc import imresize
from skimage.transform import resize
import yaml

import pycocotools.mask as mask_util
import cv2

from skimage.color import rgb2grey
from data.base_dataset import BaseDataset


class IntPhysRealDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        train_path = "/data/vision/billf/scratch/yilundu/dataset/intphys/train"
        test_path = "/data/vision/billf/scratch/jerrymei/newIntPhys/render/output/test"

        if opt.isTrain:
            p = train_path
        else:
            p = test_path

        dirs = os.listdir(p)
        self.depth = opt.depth
        files = []
        depth_files = []
        mask_files = []

        for d in dirs:
            base_path = osp.join(p, d, 'scene')
            depth_path = osp.join(p, d, 'depth')
            mask_path = osp.join(p, d, 'masks')
            ims = os.listdir(base_path)
            ims = sorted(ims)
            ims = ims

            files.append([osp.join(base_path, im) for im in ims])

            depth_ims = os.listdir(depth_path)
            depth_ims = sorted(depth_ims)
            depth_files.append([osp.join(depth_path, im) for im in depth_ims])

            mask_ims = os.listdir(mask_path)
            mask_ims = sorted(mask_ims)
            mask_files.append([osp.join(mask_path, im) for im in mask_ims])

        self.opt = opt
        self.A_paths = files
        self.D_paths = depth_files
        self.M_paths = mask_files
        self.frames = self.opt.frames
        self.depth = self.opt.depth
        self.physics_loss = self.opt.physics_loss
        self.eval_intphys = self.opt.eval_intphys
        self.optical_flow = self.opt.optical_flow

        if self.physics_loss:
            self.frames = 3


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        D_path = self.D_paths[index]
        M_path = self.M_paths[index]

        if self.optical_flow:
            offset = 1
        else:
            offset = 0

        if self.physics_loss:
            # When computing physics, forward predictions by 5 steps
            ix = random.randint(offset, len(A_path) - 9)
        else:
            ix = random.randint(offset, len(A_path) - self.frames)

        A_imgs = []
        seg_ims = []

        if self.eval_intphys:

            ims_list = []
            seg_ims_list = []
            base_ims_list = []

            for i in range(5):
                ix = random.randint(1, len(A_path)-1)
                path = A_path[ix]
                before_path = A_path[ix-1]

                labels = np.zeros((256, 256))

                if self.opt.no_multiscale_baseline:
                    im = imread(path)[:, :, :3]
                    im = resize(im, (256, 256))
                    ims = torch.Tensor(im).permute(2, 0, 1)[None, :, :, :]
                else:
                    im = im_after = imread(path)[:, :, :3]
                    im = resize(im, (1024, 1024))

                    if self.optical_flow:
                        im_before = imread(before_path)[:, :, :3]
                        im_before_grey = rgb2grey(im_before)
                        im_after_grey = rgb2grey(im_after)
                        flow_im = cv2.calcOpticalFlowFarneback(im_before_grey, im_after_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        flow_im = resize(flow_im, (1024, 1024))
                        im = np.concatenate([im, flow_im], axis=2)

                    ims = []

                    for j in range(7):
                        for k in range(7):
                            ims.append(torch.Tensor(im[j*128:j*128+256, k*128:k*128+256]).permute(2, 0, 1))

                    ims = torch.stack(ims, dim=0)

                ims_list.append(ims)
                seg_ims_list.append(torch.Tensor(labels))
                base_ims_list.append(torch.Tensor(im))

            ims_list = torch.stack(ims_list, dim=0)
            seg_ims_list = torch.stack(seg_ims_list, dim=0)
            base_ims_list = torch.stack(base_ims_list, dim=0)

            return ims_list, seg_ims_list, base_ims_list

        if self.physics_loss:
            frame_ims = []
            j = random.randint(0, 6)
            k = random.randint(0, 6)

            for i in range(self.frames):
                path = A_path[ix + i* 3]
                im = imread(path)[:, :, :3]
                im = resize(im, (1024, 1024))
                frame_ims.append(torch.Tensor(im[j*128:j*128+256, k*128:k*128+256]).permute(2, 0, 1))

            frame_ims = torch.stack(frame_ims, dim=0)
            return frame_ims

        else:
            for i in range(self.frames):

                # if self.depth:
                #     D_img = imread(D_path[ix+i])[:, :, :3]
                #     D_img = imresize(D_img, (1024, 1024))
                #     D_img = D_img.transpose((2, 0, 1)) / 255
                #     D_img = torch.from_numpy(D_img).float()
                #     A_imgs.append(D_img)
                # else:
                path_before = A_path[ix+i-1]
                path = A_path[ix+i]
                split_path = path.split("/")
                prefix_folder = split_path[-3]
                split_file = split_path[-1]
                seg_path = osp.join(self.refine_path, prefix_folder, split_file)

                if osp.exists(seg_path):
                    seg_im = imread(seg_path)
                    seg_im = resize(seg_im, (1024, 1024))
                    unique_val = np.unique(seg_im)
                    unique = np.random.choice(unique_val)
                    seg_im = (seg_im == unique)
                else:
                    seg_im = np.zeros((1024, 1024))

                A_img = A_img_after = imread(A_path[ix+i])[:, :, :3]
                A_before = imread(A_path[ix+i-1])[:, :, :3]

                if self.opt.no_multiscale_baseline:
                    A_img  = imresize(A_img, (256, 256))
                    A_img = A_img.transpose((2, 0, 1)) / 255.
                    A_img = np.clip(A_img + np.random.uniform(-1/512, 1/512, A_img.shape), 0, 1)
                    A_img = torch.Tensor(A_img).float()
                    seg_im = torch.Tensor(resize(seg_im, (256, 256)))
                else:
                    A_img = imresize(A_img, (1024, 1024))

                    if self.optical_flow:
                        A_before_grey = rgb2grey(A_before)
                        A_after_grey = rgb2grey(A_img_after)
                        flow_im = cv2.calcOpticalFlowFarneback(A_before_grey, A_after_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        flow_im = resize(flow_im, (1024, 1024))

                        A_img = np.concatenate([A_img, flow_im], axis=2)

                    x = random.randint(0, 6)
                    y = random.randint(0, 6)

                    A_img = A_img[x*128:x*128+256, y*128:y*128+256]
                    seg_im = seg_im[x*128:x*128+256, y*128:y*128+256]
                    seg_im[seg_im > 0] = 1

                    # A_img = imresize(A_img, (128, 128))
                    A_img = A_img.transpose((2, 0, 1)) / 255.
                    A_img = np.clip(A_img + np.random.uniform(-1/512, 1/512, A_img.shape), 0, 1)
                    A_img = torch.from_numpy(A_img).float()

                    if self.depth:
                        d_img = imread(D_path[ix+i])[:, :, :3]
                        d_img = resize(d_img, (1024, 1024))
                        d_img = d_img[x*128:x*128+256, y*128:y*128+256, 0:1].transpose((2, 0, 1))
                        A_img = torch.cat([A_img, torch.Tensor(d_img)], dim=0)

                A_imgs.append(A_img)
                seg_ims.append(torch.Tensor(seg_im))

            A_img = torch.cat(A_imgs, dim=0)
            seg_ims = torch.cat(seg_ims, dim=0)
            return A_img, seg_ims

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
