import os
import os.path as osp
import numpy as np
import cv2

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
# import cv2

from skimage.color import rgb2grey


def filter_labels(labels):
    label_ids = np.unique(labels)
    labels_new = np.zeros_like(labels)

    y, x = labels.shape

    bboxs = []

    for label_id in label_ids:
        if label_id == 0:
            continue
        else:
            label_mask = (labels == label_id)
            label_idx = np.arange(label_mask.size)[label_mask.flatten()]
            res_x = label_idx % x
            res_y = (label_idx / x).astype(np.int32)
            x_min, x_max = res_x.min(), res_x.max()
            y_min, y_max = res_y.min(), res_y.max()

            bboxs.append((x_min, x_max, y_min, y_max, label_id))

    for i in range(len(bboxs)):
        select_bboxs = bboxs[:i] + bboxs[i+1:]
        bbox = bboxs[i]

        valid = False

        for bbox_i in select_bboxs:
            sx_min, sx_max, sy_min, sy_max, label_val = bbox_i
            x_min, x_max, y_min, y_max, _ = bbox

            if sy_max > y_min and y_max > sy_min:
                if sx_max > x_min and x_max > sx_min:
                    valid = True
                    break

        if valid:
            labels_new = labels_new + (labels == label_val) * label_val

    return labels_new


class IntPhysDataset(data.Dataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        if opt.intphys_multicolor:
            train_path = "/data/vision/billf/scratch/jerrymei/projects/newIntPhys/render/output/train_v11"
        elif opt.human:
            train_path = "/data/vision/billf/scratch/yilundu/data/human"
        else:
            train_path = "/data/vision/billf/scratch/yilundu/data/train_v7"

        test_path = "/data/vision/billf/scratch/yilundu/data/human"

        p = train_path

        self.refine_path = "/data/vision/billf/scratch/yilundu/adept/ADEPT-Model-Release/motion_annotation"
        self.refine_motion = opt.refine_motion
        self.eval_intersection = opt.eval_intersection
        dirs = os.listdir(p)
        self.depth = opt.depth
        files = []
        depth_files = []

        for d in dirs:
            base_path = osp.join(p, d, 'imgs')
            depth_path = osp.join(p, d, 'depths')
            ims = os.listdir(base_path)
            ims = sorted(ims)
            ims = ims

            # depth_ims = depth_ims[::5]

            files.append([osp.join(base_path, im) for im in ims])

            if self.depth:
                depth_ims = os.listdir(depth_path)
                depth_ims = sorted(depth_ims)
                assert (len(ims) == len(depth_ims))
                depth_files.append([osp.join(depth_path, im) for im in depth_ims])

        self.opt = opt
        self.A_paths = files
        self.D_paths = depth_files
        self.frames = self.opt.frames
        self.depth = self.opt.depth
        self.physics_loss = self.opt.physics_loss
        self.eval_intphys = self.opt.eval_intphys
        self.optical_flow = self.opt.optical_flow

        self.im_size = 256

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

        if self.eval_intphys:
            split_path = A_path[0].split("/")[:-1]
            base_path = osp.join(*split_path[:-1])
            yaml_file = osp.join("/" + base_path, "{}_ann.yaml".format(split_path[-2]))
            data = yaml.safe_load(open(yaml_file, "r"))
            scene = data['scene']

        if self.depth:
            D_path = self.D_paths[index]

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

        half_size = self.im_size // 2
        im_size = full_size = self.im_size

        if self.eval_intphys:

            ims_list = []
            seg_ims_list = []
            base_ims_list = []
            seg_id = []

            for i in range(5):
                ix = random.randint(1, len(A_path)-1)
                path = A_path[ix]
                before_path = A_path[ix-1]
                im_info = scene[ix]['objects']
                labels = np.zeros((320, 480))

                counter = 1
                seg_i = {}

                for ob in im_info:
                    mask_i = mask_util.decode(ob['mask'])
                    labels = labels + mask_i * counter * (labels == 0).astype(np.float32)
                    seg_i[counter] = ob['name']
                    counter = counter + 1

                seg_id.append(seg_i)

                labels = filter_labels(labels)

                if self.opt.no_multiscale_baseline:
                    im = imread(path)[:, :, :3]
                    im = resize(im, (self.im_size, self.im_size))
                    ims = torch.Tensor(im).permute(2, 0, 1)[None, :, :, :]
                else:
                    im = im_after = imread(path)[:, :, :3]
                    im = resize(im, (4 * self.im_size, 4 * self.im_size))

                    if self.optical_flow:
                        im_before = imread(before_path)[:, :, :3]
                        im_before_grey = rgb2grey(im_before)
                        im_after_grey = rgb2grey(im_after)
                        flow_im = cv2.calcOpticalFlowFarneback(im_before_grey, im_after_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        flow_im = (flow_im - flow_im.min()) / (flow_im.max() - flow_im.min() + 1e-20)
                        flow_im = resize(flow_im, (4 * self.im_size, 4 * self.im_size))
                        im = np.concatenate([im, flow_im], axis=2)

                    ims = []

                    for j in range(7):
                        for k in range(7):
                            ims.append(torch.Tensor(im[j*half_size:j*half_size+im_size, k*half_size:k*half_size+im_size]).permute(2, 0, 1))

                    ims = torch.stack(ims, dim=0)

                ims_list.append(ims)
                seg_ims_list.append(torch.Tensor(labels))
                base_ims_list.append(torch.Tensor(im))

            ims_list = torch.stack(ims_list, dim=0)
            seg_ims_list = torch.stack(seg_ims_list, dim=0)
            base_ims_list = torch.stack(base_ims_list, dim=0)

            return ims_list, seg_ims_list, base_ims_list, seg_id

        if self.physics_loss:
            frame_ims = []
            j = random.randint(0, 6)
            k = random.randint(0, 6)

            for i in range(self.frames):
                path = A_path[ix + i* 3]
                before_path = A_path[ix + i* 3 - 1]
                if self.opt.no_multiscale_baseline:
                    im = imread(path)[:, :, :3]
                    im = resize(im, (self.im_size, self.im_size))
                    frame_ims.append(torch.Tensor(im).permute(2, 0, 1))
                else:
                    im = im_after = imread(path)[:, :, :3]
                    im = resize(im, (4*self.im_size, 4*self.im_size))

                    if self.optical_flow:
                        im_before = imread(before_path)[:, :, :3]
                        im_before_grey = rgb2grey(im_before)
                        im_after_grey = rgb2grey(im_after)

                        flow_im = cv2.calcOpticalFlowFarneback(im_before_grey, im_after_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        flow_im = (flow_im - flow_im.min()) / (flow_im.max() - flow_im.min() + 1e-20)
                        flow_im = resize(flow_im, (4*self.im_size, 4*self.im_size))
                        im = np.concatenate([im, flow_im], axis=2)

                    frame_ims.append(torch.Tensor(im[j*half_size:j*half_size+full_size, k*half_size:k*half_size+full_size]).permute(2, 0, 1))

            frame_ims = torch.stack(frame_ims, dim=0)
            return frame_ims

        else:
            x = random.randint(0, 6)
            y = random.randint(0, 6)

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
                    seg_im = resize(seg_im, (4*full_size, 4*full_size))
                    unique_val = np.unique(seg_im)
                    unique = np.random.choice(unique_val)
                    seg_im = (seg_im == unique)
                else:
                    seg_im = np.zeros((4*full_size, 4*full_size))

                A_img = A_img_after = imread(A_path[ix+i])[:, :, :3]
                A_before = imread(A_path[ix+i-1])[:, :, :3]

                if self.opt.no_multiscale_baseline:
                    A_img  = imresize(A_img, (full_size, full_size))
                    A_img = A_img.transpose((2, 0, 1)) / 255.
                    A_img = np.clip(A_img + np.random.uniform(-1/512, 1/512, A_img.shape), 0, 1)
                    A_img = torch.Tensor(A_img).float()
                    seg_im = torch.Tensor(resize(seg_im, (full_size, full_size)))
                else:
                    A_img = imresize(A_img, (4*full_size, 4*full_size))

                    if self.optical_flow:
                        A_before_grey = rgb2grey(A_before)
                        A_after_grey = rgb2grey(A_img_after)
                        flow_im = cv2.calcOpticalFlowFarneback(A_before_grey, A_after_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        flow_im = (flow_im - flow_im.min()) / (flow_im.max() - flow_im.min() + 1e-20)
                        flow_im = resize(flow_im, (4*full_size, 4*full_size))
                        # print("A_img bounds: ", A_img.min(), A_img.max())
                        # print("flow_im bounds: ", flow_im.min(), flow_im.max())
                        # flow_im_save = np.concatenate([flow_im, np.zeros_like(flow_im[:, :, 0:1])], axis=2)
                        # imsave("flow.png", flow_im_save)



                        A_img = np.concatenate([A_img, flow_im * 255], axis=2)


                    A_img = A_img[x*half_size:x*half_size+full_size, y*half_size:y*half_size+full_size]
                    seg_im = seg_im[x*half_size:x*half_size+full_size, y*half_size:y*half_size+full_size]
                    seg_im[seg_im > 0] = 1

                    # A_img = imresize(A_img, (128, 128))
                    A_img = A_img.transpose((2, 0, 1)) / 255.
                    A_img = np.clip(A_img + np.random.uniform(-1/512, 1/512, A_img.shape), 0, 1)
                    A_img = torch.from_numpy(A_img).float()

                    if self.depth:
                        d_img = imread(D_path[ix+i])[:, :, :3]
                        d_img = resize(d_img, (4*full_size, 4*full_size))
                        d_img = d_img[x*half_size:x*half_size+full_size, y*half_size:y*half_size+full_size, 0:1].transpose((2, 0, 1))
                        A_img = torch.cat([A_img, torch.Tensor(d_img)], dim=0)

                A_imgs.append(A_img)
                seg_ims.append(torch.Tensor(seg_im))

            A_img = torch.cat(A_imgs, dim=0)
            seg_ims = torch.cat(seg_ims, dim=0)
            return A_img, seg_ims

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
