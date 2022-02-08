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


def construct_seg_mask(m_k, debug=False):
    batch_size, timestep = m_k.shape[0], m_k.shape[1]

    seg_im = np.zeros((1024, 1024))
    count = 1
    for k in range(49):
        x = k // 7
        y = k % 7
        for n in range(1, 4):
            mask_instance = (m_k[k, n] > 0.03)

            # Only take the largest connected component of a mask
            # TODO this may be relaxed to component > then x size
            instance_map = measure.label(mask_instance * 255)
            instance_val, counts = np.unique(instance_map, return_counts=True)

            counts_idx = np.argsort(counts)
            sort_instance = instance_val[counts_idx]
            sort_count = counts[counts_idx]

            mask_instance = (instance_map == -2)

            for o in range(1, sort_count.shape[0]+1):
                if sort_instance[-o] == 0:
                    continue

                if sort_count[-o] < 30:
                    break

                mask_instance_i = (instance_map == sort_instance[-o])
                mask_instance = (mask_instance | mask_instance_i)

            select_seg = seg_im[x*128:x*128+256, y*128:y*128+256]
            overlap = select_seg[mask_instance].sum()

            if overlap > 0:
                val, counts = np.unique(select_seg[mask_instance], return_counts=True)
                sort_idx = np.argsort(counts)
                mask_val = val[sort_idx[-1]]
                if mask_val == 0:
                    mask_val = val[sort_idx[-2]]
            else:
                if mask_instance.sum() < 100:
                    continue

                mask_val = count
                count = count + 1

            mask_instance = mask_instance.astype(np.float32) * mask_val
            select_seg[select_seg == 0] = mask_instance[select_seg == 0]

    return seg_im


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    data = next(dataset)

    for i in range(data[0].size(0)):
        inp = data[0][i]
        rgb_im = data[1][i]
        model.set_input((inp, inp))         # unpack data from dataset and apply preprocessing
        model.forward()
        m = model.m.detach().cpu().numpy()
        mask = construct_seg_mask(m)
        panel_im = np.concatenate([np.tile(mask[:, :, None], (1, 1, 3)), rgb_im], axis=0)
        imwrite("robotnet_mask_{}.png".format(i), panel_im)
        print("here")
