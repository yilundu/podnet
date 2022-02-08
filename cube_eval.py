"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np
from skimage import measure
from skimage import filters, color
from skimage.segmentation import slic
from skimage.future import graph
from skimage.transform import resize
from imageio import imwrite
import math
import torch
import os.path as osp
import random
import matplotlib._color_data as mcd
from webcolors import hex_to_rgb

color_dict = mcd.CSS4_COLORS
hex_val = list(color_dict.values())


if __name__ == '__main__':
    from scipy.misc import imsave
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    dataset = dataset.dataset

    inps = []
    masks = []

    for i in range(32):
        im, _ = dataset[i]
        inps.append(resize(im.permute(1, 2, 0), (64, 64)))
        inp = im[None, :].to(model.device)

        with torch.no_grad():
            s = inp.size()
            print(s)
            model.set_input((inp, inp))         # unpack data from dataset and apply preprocessing
            model.forward()
            m = model.m.detach().cpu().numpy()[0]
            m = [resize(mi, (64, 64)) for mi in m]
            masks.append(np.array(m))

    inps = np.array(inps)
    s = inps.shape
    inps = inps.reshape(s[0] * s[1], s[2], 3)
    imsave("inp.png", inps)

    masks = np.array(masks)
    s = masks.shape
    masks[:, :, :1, :] = 1
    masks[:, :, -1:, :] = 1
    masks[:, :, :, :1] = 1
    masks[:, :, :, -1:] = 1
    masks = masks.transpose((0, 2, 1, 3)).reshape(s[0] * s[2], s[1] * s[3])
    imsave("masks.png", masks)
