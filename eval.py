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
from skimage import data, segmentation, color
from scipy.misc import imresize, imread
import pybullet as p
from pybullet_utils import set_pose, Pose, get_pose, enable_gravity, load_model, get_aabb, create_body
p.connect(p.DIRECT)

color_dict = mcd.CSS4_COLORS
hex_val = list(color_dict.values())

_shape_net_names = ['airplane', 'ashcan', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'bicycle', 'birdhouse',
                    'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'chair',
                    'computer_keyboard', 'dishwasher', 'display', 'earphone', 'faucet', 'file', 'guitar', 'helmet',
                    'jar', 'knife', 'lamp', 'laptop', 'loudspeaker', 'mailbox', 'microphone', 'microwave', 'motorbike',
                    'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote', 'rifle', 'rocket', 'skateboard',
                    'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'vessel', 'washer', 'wine_bottle']


def is_stable(cubes):
    p.resetSimulation()

    ids = []

    for cube in cubes:
        center = np.array(cube['center'])
        scale = [cube['scale'][0], 0.1, cube['scale'][1]]
        vid = p.createVisualShape(p.GEOM_BOX, halfExtents=list(scale))
        cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(scale))
        # obj = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=cid, baseVisualShapeIndex=vid)
        obj = create_body(cid, vid, mass=1.0)
        pose = Pose(point=[center[0], 0, center[1]])
        set_pose(obj, pose)
        aabb = get_aabb(obj)
        ids.append(obj)

    load_model("models/short_floor.urdf", scale=20.0, fixed_base=True)
    enable_gravity()
    viewMatrix = p.computeViewMatrix(cameraEyePosition=[0.5, 1.0, 1.0], cameraTargetPosition=[0.0, 0.0, 0.0], cameraUpVector=[0.0, 0.0, 1.0])
    projectionMatrix = p.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.02, farVal=50.0)

    for i in range(100):
        p.stepSimulation()

    state = []
    for id in ids:
        pose = np.array(get_pose(id)[0])
        state.append(pose)

    state = np.array(state)

    for i in range(100):
        p.stepSimulation()


        _, _, im, _ , _ = p.getCameraImage(128, 128, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix, renderer=p.ER_TINY_RENDERER)
        # imwrite("im{}.png".format(i), im)

    new_state = []
    for id in ids:
        pose = np.array(get_pose(id)[0])
        new_state.append(pose)

    new_state = np.array(new_state)
    dist = np.abs(new_state - state).mean()
    print(dist)

    return (dist > 1e-3)



def construct_no_multiscale_seg_mask(m_k, mask_tresh):
    m_k = m_k[0]
    masks = np.zeros((256, 256))

    counter = 1

    for i in [1, 2, 3]:
        mask_i = m_k[i]
        mask_filter = (mask_i > mask_tresh)

        if mask_filter.astype(np.float32).sum() > 0:
            masks[mask_filter] = np.ones(masks.shape)[mask_filter] * counter
            counter = counter + 1

    return masks

def make_color_mask(m_k):
    color_m_k = np.zeros((m_k.shape[0], m_k.shape[1], 3))
    ixs = np.unique(m_k)

    for ix in ixs:

        if ix == 0:
            continue
        else:
            seg_mask = (m_k == ix).astype(np.float32)
            ix = int(ix)
            rgb = hex_to_rgb(hex_val[ix])
            pixel = np.array([rgb.red, rgb.green, rgb.blue])[None, None, :]

            color_m_k = (1 - seg_mask[:, :, None]) * color_m_k + seg_mask[:, :, None] * pixel

    return color_m_k



def construct_seg_mask(m_k, max_slot, mask_tresh, debug=False):
    batch_size, timestep = m_k.shape[0], m_k.shape[1]

    seg_im = np.zeros((1024, 1024))
    count = 1
    for k in range(49):
        x = k // 7
        y = k % 7
        for n in range(1, max_slot):
            # For intphys it is 0.03
            # For cubes is 0.9
            mask_instance = (m_k[k, n] > mask_tresh)
            # mask_instance = (m_k[k, n] > 0.9)

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

                # if sort_count[-o] < 1:
                if sort_count[-o] < 1:
                    break

                mask_instance_i = (instance_map == sort_instance[-o])
                mask_instance = (mask_instance | mask_instance_i)

            select_seg = seg_im[x*128:x*128+256, y*128:y*128+256]
            overlap = select_seg[mask_instance].sum()

            if overlap > 20:
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

def compute_mean_iou(pred_mask, gt_mask, seg_id, stat_dict):
    gt_answers = np.unique(gt_mask)
    mask_idx = np.unique(pred_mask)
    pred_mask_expand = np.zeros((mask_idx.shape[0] - 1, gt_mask.shape[0], gt_mask.shape[1]))

    counter = 0
    for m in mask_idx:
        if m == 0:
            continue

        pred_mask_expand[counter] = resize((pred_mask == m).astype(np.float32), (gt_mask.shape[0], gt_mask.shape[1]))
        counter = counter + 1

    # pred_mask_expand = (pred_mask_expand > 0.8)
    pred_mask_expand = (pred_mask_expand > 0.2)
    ious = []
    accs = []
    for idx in gt_answers:
        if idx == 0:
            continue

        gt_mask_i = (gt_mask == idx)[None, :, :]

        union = (pred_mask_expand | gt_mask_i).astype(np.float32)
        intersection = (pred_mask_expand & gt_mask_i).astype(np.float32)
        iou = intersection.sum(axis=1).sum(axis=1) / union.sum(axis=1).sum(axis=1)



        # for i in range(union.shape[0]):
        #     imwrite("intersection_{}.png".format(i), intersection[i])
        #     imwrite("union_{}.png".format(i), union[i])

        if iou.size > 0:
            iou = iou.max()

            # if idx in seg_id:
            #     cls = seg_id[idx]

            #     if cls[:4] != 'cube':
            #         idx = int(cls[:4])
            #         name = _shape_net_names[idx]
            #         if name in stat_dict:
            #             stat_dict[name].append(iou)
            #         else:
            #             stat_dict[name] = [iou]

            if iou >= 0.5:
                accs.append(1)
            else:
                accs.append(0)

            ious.append(iou)

    if len(ious) > 0:
        return np.mean(ious), np.mean(accs)
    else:
        return None, None




if __name__ == '__main__':
    from scipy.misc import imsave
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    if opt.dataset_mode == 'block':
        max_slot = opt.num_slots
        mask_tresh = 0.9
    else:
        max_slot = opt.num_slots - 1
        # mask_tresh = 0.03
        if opt.optical_flow:
            mask_tresh = 0.2
        elif opt.monet_baseline:
            mask_tresh = 0.2
        else:
            # mask_tresh = 0.9
            mask_tresh = 0.9

    ious = []
    accs = []
    counter = 0

    # torch.manual_seed(0)
    # random.seed(0)

    stat_dict = {}
    corrects = []
    for data in dataset.dataloader:
        inp, gt_mask, orig_img, stability = data
        s = inp.size()
        inp, gt_mask, orig_img = inp.view(-1, *inp.size()[2:]), gt_mask.view(-1, *gt_mask.size()[2:]), orig_img.view(-1, *orig_img.size()[2:])

        data = (inp, gt_mask, orig_img, stability)

        chunk = 2


        with torch.no_grad():
            for i in range(0, data[0].size(0), chunk):
                index = i
                inp = data[0][i:i+chunk]
                gt_mask = data[1][i:i+chunk]
                orig_img = data[2][i:i+chunk]
                seg_id = data[3][i:i+chunk]

                s = inp.size()
                chunk_iter = s[0]
                inp = inp.view(-1, *s[2:])
                model.set_input((inp, inp))         # unpack data from dataset and apply preprocessing
                model.forward()
                m = model.m.detach().cpu().numpy()
                m = m.reshape(chunk_iter, -1, *m.shape[1:])

                for j in range(chunk_iter):

                    if opt.no_multiscale_baseline:
                        mask = construct_no_multiscale_seg_mask(m[j], mask_tresh)
                    else:
                        mask = construct_seg_mask(m[j], max_slot=max_slot, mask_tresh=mask_tresh)

                    idx, vals = np.unique(mask, return_counts=True)
                    sort_idx = np.argsort(vals)[::-1]
                    idx = idx[sort_idx]
                    vals = vals[sort_idx]

                    obj_coords = []
                    bbs = []

                    global_y_min = float('inf')
                    for ix, count in zip(idx[1:], vals[1:]):

                        if count < 10000:
                            break

                        obj_coord = []
                        mask_cube = (mask == ix)
                        label_mask = np.arange(mask.size)[mask_cube.flatten()]

                        res_x = (label_mask % 1024)
                        res_y = (label_mask / 1024).astype(np.int32)
                        x_min, x_max = res_x.min(), res_x.max()
                        y_min, y_max = res_y.min(), res_y.max()
                        x_min, x_max = x_min / 1024., x_max / 1024.
                        y_min, y_max = y_min / 1024., y_max / 1024.
                        global_y_min = min(global_y_min, y_min)

                    for ix, count in zip(idx[1:], vals[1:]):

                        if count < 10000:
                            break

                        obj_coord = []
                        mask_cube = (mask == ix)
                        label_mask = np.arange(mask.size)[mask_cube.flatten()]

                        res_x = (label_mask % 1024)
                        res_y = (label_mask / 1024).astype(np.int32)
                        x_min, x_max = res_x.min(), res_x.max()
                        y_min, y_max = res_y.min(), res_y.max()
                        x_min, x_max = x_min / 1024., x_max / 1024.
                        y_min, y_max = y_min / 1024., y_max / 1024.

                        bbs.append({'center': [(x_min + x_max) / 2., (y_min + y_max) / 2. - global_y_min], 'scale': [(x_max - x_min) / 2., (y_max - y_min) / 2.]})

                    # imsave("mask.png", mask)
                    # stable_val = is_stable(bbs)
                    # stable_label = stability[index+j]
                    # corrects.append(stable_val == stable_label)
                    # print(np.mean(corrects))

                    # base_path = "output"
                    # scale = 255 / mask.max()
                    # mask_tile = np.tile(mask[:, :, None], (1, 1, 3)) * scale
                    # mask_tile = make_color_mask(mask)
                    # im = resize(orig_img[j].cpu().detach().numpy().transpose((1, 2, 0)), (mask.shape[0], mask.shape[1])) * 255
                    # im = orig_img[j] * 255

                   #  panel_im = np.concatenate([mask_tile, im], axis=0)

                   #  imsave(osp.join(base_path, "panel_{}.png".format(counter)), panel_im)

                   #  counter += 1
                   #  if counter > 100:
                   #      assert False

                    # imsave("gt_mask.png", gt_mask.numpy().astype(np.float32))
                    # img = orig_img[j].cpu().detach().numpy()
                    # img = img.astype(np.float64)
                    # imsave("img{}.png".format(counter), img)
                    # # assert False
                    # # img = imread("im0.png")
                    # img = imresize(img, (gt_mask[j].shape[0], gt_mask[j].shape[1]))
                    # labels1 = segmentation.slic(img, compactness=1.0, n_segments=500)

                    # g = graph.rag_mean_color(img, labels1, mode='similarity')
                    # mask = graph.cut_normalized(labels1, g)
                    # out1 = color.label2rgb(mask, img, kind='avg', bg_label=0)
                    # imwrite("mask{}.png".format(counter), out1)
                    # counter = counter + 1
                    # assert False

                    # import pdb
                    # pdb.set_trace()
                    # print("here")
                    # print(inp)
                    # print(orig_img)
                    iou, acc = compute_mean_iou(mask, gt_mask[j].numpy(), 0, stat_dict)

                    if iou is not None:
                        ious.append(iou)

                    if acc is not None:
                        accs.append(acc)

                    print("current mean iou value ", np.mean(ious), np.std(ious) / (math.pow(len(ious), 0.5)))
                    print("current mean acc value ", np.mean(accs), np.std(accs) / (math.pow(len(accs), 0.5)))

                print("current key average: ")

                for k, v in stat_dict.items():
                    print("{}: {}".format(k, np.mean(v)))
