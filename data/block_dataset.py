import os
import os.path as osp
import numpy as np

import torchvision.transforms.functional as TF
import random

from PIL import Image
import torch.utils.data as data
from scipy.misc import imread
import torch

from scipy.misc import imresize
from skimage.transform import resize
from data.base_dataset import BaseDataset


class BlockDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.eval_cube = opt.eval_cube

        if self.eval_cube:
            p = "/data/vision/billf/object-properties/dataset/billf-6/physnet/real"
        else:
            p = "/data/vision/billf/object-properties/dataset/billf-6/physnet/real_frames"

        self.label = np.loadtxt("/data/vision/billf/object-properties/dataset/billf-6/physnet/real.txt")
        folders = os.listdir(p)
        paths = []

        if self.eval_cube:
            for folder in folders:
                base_folder = osp.join(p, folder)
                ims = os.listdir(base_folder)
                ims_paths = []
                seg_paths = []

                for im in ims:
                    if "mask.png" in im:
                        seg_paths.append(osp.join(base_folder, im))
                    elif "rgb.png" in im:
                        ims_paths.append(osp.join(base_folder, im))

                seg_paths = sorted(seg_paths)
                ims_paths = sorted(ims_paths)
                joint_paths = list(zip(seg_paths, ims_paths))
                paths.append(joint_paths)
        else:
            for folder in folders:
                base_folder = osp.join(p, folder)
                ims = os.listdir(base_folder)
                ims_paths = []

                for im in ims:
                    if ".mp4" in im:
                        pass
                    else:
                        ims_paths.append(osp.join(base_folder, im))

                paths.append(ims_paths)

        self.opt = opt
        self.paths = paths
        self.frames = self.opt.frames
        self.physics_loss = self.opt.physics_loss

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
        im_path = self.paths[index]

        if self.physics_loss:
            # When computing physics, forward predictions by 5 steps
            ix = random.randint(min(10, len(im_path)-4), len(im_path) - 3)
        else:
            ix = random.randint(0, len(im_path) - self.frames)

        total_im = []
        if self.eval_cube:
            ix = 0

            ims_list = []
            seg_ims_list = []
            base_ims_list = []
            # for i in range(len(im_path)):
            for i in range(1):
                seg_path, rgb_path = im_path[i]
                im = imread(rgb_path)
                seg_im = imread(seg_path)

                if self.opt.no_multiscale_baseline:
                    im = torch.Tensor(resize(im, (256, 256)).transpose((2, 0, 1)))
                    ims = [im]
                    ims = torch.stack(ims, dim=0)
                else:
                    im = resize(im, (1024, 1024))
                    ims = []


                    for j in range(7):
                        for k in range(7):
                            ims.append(torch.Tensor(im[j*128:j*128+256, k*128:k*128+256]).permute(2, 0, 1))

                    ims = torch.stack(ims, dim=0)

                ims_list.append(ims)
                seg_ims_list.append(torch.Tensor(seg_im))
                base_ims_list.append(torch.Tensor(im))

            ims_list = torch.stack(ims_list, dim=0)
            seg_ims_list = torch.stack(seg_ims_list, dim=0)
            base_ims_list = torch.stack(base_ims_list, dim=0)
            stability = self.label[index, -1]

            return ims_list, seg_ims_list, base_ims_list, [stability]


        if self.physics_loss:
            frame_ims = []
            j = random.randint(0, 6)
            k = random.randint(0, 6)

            for i in range(self.frames):
                im = imread(im_path[ix+i])

                if self.opt.no_multiscale_baseline:
                    im = resize(im, (256, 256))
                    frame_ims.append(torch.Tensor(im).permute(2, 0, 1))
                else:
                    im = resize(im, (1024, 1024))
                    frame_ims.append(torch.Tensor(im[j*128:j*128+256, k*128:k*128+256]).permute(2, 0, 1))

            frame_ims = torch.stack(frame_ims, dim=0)
            return frame_ims
        else:
            for i in range(self.frames):
                if self.opt.no_multiscale_baseline:
                    A_img = imread(im_path[ix+i])
                    A_img  = imresize(A_img, (256, 256))
                    A_img = A_img.transpose((2, 0, 1)) / 255.
                    A_img = np.clip(A_img + np.random.uniform(-1/512, 1/512, A_img.shape), 0, 1)
                    im = torch.Tensor(A_img).float()
                else:
                    im = imread(im_path[ix+i])
                    im = resize(im, (1024, 1024))

                    i = random.randint(0, 6)
                    j = random.randint(0, 6)

                    select_im = im[i*128:i*128+256, j*128:j*128+256]
                    im = torch.Tensor(select_im).permute(2, 0, 1)
                total_im.append(im)

            im = torch.cat(total_im, dim=0)

            return im, 0

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.paths)
