import tensorflow as tf
from robonet.datasets.robonet_dataset import RoboNetDataset
from robonet.datasets import load_metadata
from data.base_dataset import BaseDataset
import numpy as np
import random
import torch

class RobotNetDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        all_robonet = load_metadata("/data/vision/billf/scratch/yilundu/robonet/hdf5")
        sess = tf.InteractiveSession()
        database = all_robonet[all_robonet['adim']==4]
        self.database = database

        data = RoboNetDataset(batch_size=opt.batch_size, dataset_files_or_metadata=database, hparams={'img_size': [1024, 1024], 'load_T': 2, 'target_adim':4, 'action_mismatch':1})
        self.data = data
        self.images = data['images']
        self.sess = sess
        self.dataloader = self
        self.full_robonet = opt.full_robonet


    def __next__(self):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        images = self.sess.run(self.images)
        images = images[:, 0, 0]
        images = np.clip(images + np.random.uniform(0, 1/256, images.shape), 0, 1)
        batch = images.shape[0]

        if self.full_robonet:
            panel_im = np.zeros((batch, 49, 256, 256, 3))
            for idx, im in enumerate(images):
                for i in range(7):
                    for j in range(7):
                        select_idx = i * 7 + j
                        select_im = im[i*128:i*128+256, j*128:j*128+256]
                        panel_im[idx, select_idx] = select_im

            panel_im = torch.FloatTensor(panel_im)
            panel_im = panel_im.permute(0, 1, 4, 2, 3)

            return panel_im, images
        else:
            panel_im = np.zeros((batch, 256, 256, 3))
            for idx, im in enumerate(images):
                i, j = random.randint(0, 6), random.randint(0, 6)
                select_im = im[i*128:i*128+256, j*128:j*128+256]
                panel_im[idx] = select_im

            panel_im = torch.FloatTensor(panel_im)
            panel_im = panel_im.permute(0, 3, 1, 2)

            return panel_im, panel_im

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        pass

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.database)

if __name__ == "__main__":
    # sess = tf.InteractiveSession()
    # all_robonet = load_metadata("/data/vision/billf/scratch/yilundu/robonet/hdf5")
    # database = all_robonet[all_robonet['adim']==4]
    # import pdb
    # pdb.set_trace()
    # data = RoboNetDataset(batch_size=16, dataset_files_or_metadata=database, hparams={'img_size': [1024, 1024], 'load_T': 2, 'target_adim':4, 'action_mismatch':1})
    # images = data['images']
    # real_image = sess.run(images)
    # import pdb
    # pdb.set_trace()
    # print("here")
    # assert False

    from easydict import EasyDict

    opt = EasyDict()
    opt.batch_size = 16
    dataset = RobotNetDataset(opt)
    it = iter(dataset)
    images = next(it)
    print(images.shape)

