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


class CubeDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        p = "/data/vision/billf/object-properties/yilundu/sandbox/image_comb/cubes_varied_multi_311.npz"
        self.data = np.load(p)
        self.ims = np.array(self.data['ims'])
        self.labels = np.array(self.data['labels'])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        im = self.ims[index]
        im = imresize(im, (256, 256)) / 256.
        im = torch.FloatTensor(im).permute(2, 0, 1)
        return im, torch.zeros(1)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.ims.shape[0]
