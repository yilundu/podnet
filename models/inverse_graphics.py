from torch.nn import ModuleList
import torch.nn.functional as F
import torch.nn as nn
import torch
from easydict import EasyDict
from torch.nn.utils import spectral_norm
from .resnet import resnet18, BasicBlockUpsample
from skimage.io import imsave


class Mask2Cube(nn.Module):
    def __init__(self):
        super(Mask2Cube, self).__init__()
        # First 4 coordinates are bounding boxes, the last two are the closest and further
        # in depth map
        self.resnet = resnet18(num_classes=7)


    def forward(self, x):
        output = self.resnet(x)

        return output


class Cube2Mask(nn.Module):
    def __init__(self):
        super(Cube2Mask, self).__init__()
        self.fc1 = nn.Linear(7, 512)
        self.fc2 = nn.Linear(512, 64*4*4)
        self.upsample_1 = BasicBlockUpsample(64, 64)
        self.upsample_2 = BasicBlockUpsample(64, 64)
        self.upsample_3 = BasicBlockUpsample(64, 64)
        self.upsample_4 = BasicBlockUpsample(64, 32)
        self.upsample_5 = BasicBlockUpsample(32, 32)
        self.conv = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # print(x.size())
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = h.view(h.size(0), 64, 4, 4)
        h = self.upsample_1(h)
        h = self.upsample_2(h)
        h = self.upsample_3(h)
        h = self.upsample_4(h)
        h = self.upsample_5(h)
        output = F.upsample(h, size=(256, 256))
        output = self.conv(output)
        output = F.sigmoid(output)

        return output


class Mask2CubeManual(nn.Module):
    def __init__(self):
        super(Mask2CubeManual, self).__init__()
        # First 4 coordinates are bounding boxes, the last two are the closest and further
        # in depth map
        self.resnet = resnet18(num_classes=7)


    def forward(self, x):
        assert (x.size(2) == 256)
        x_dense = x.view(-1, 256 * 256)
        x_coord = torch.arange(0, 256).to(x.device)
        y_coord, x_coord = torch.meshgrid(x_coord, x_coord)

        x_coord = x_coord.contiguous().view(256 * 256)
        y_coord = y_coord.contiguous().view(256 * 256)

        mask = (x_dense > 0.5).sum(dim=1)

        batch = x.size(0)
        mask_cpu = mask.cpu()

        prims = []

        # Assume a 60 FOV
        fx, fy = 221, 221

        for i in range(batch):
            if mask_cpu[i] > 400:
                x_dense_select_i = x_dense[i]
                mask_i = (x_dense_select_i > 0.5)

                weight_mask_i = x_dense_select_i[mask_i]
                x_coord_i = x_coord[mask_i]
                y_coord_i = y_coord[mask_i]

                # Take the top 200 indices
                x_val, idx = torch.topk(x_coord_i, 200, largest=True)
                weights_select = weight_mask_i[idx]
                x_max = (weights_select * x_val).sum(dim=0) / weights_select.sum()

                x_val, idx = torch.topk(x_coord_i, 200, largest=False)
                weights_select = weight_mask_i[idx]
                x_min = (weights_select * x_val).sum(dim=0) / weights_select.sum()

                y_val, idx = torch.topk(y_coord_i, 200, largest=True)
                weights_select = weight_mask_i[idx]
                y_max = (weights_select * y_val).sum(dim=0) / weights_select.sum()

                y_val, idx = torch.topk(y_coord_i, 200, largest=False)
                weights_select = weight_mask_i[idx]
                y_min = (weights_select * y_val).sum(dim=0) / weights_select.sum()

                y_min, y_max = 255 - y_max, 255 - y_min
                z = 1 + y_min / 128.

                x_min, x_max = (x_min - 128), (x_max - 128)
                x_3d_min, x_3d_max = x_min / fx / z, x_max / fy / z
                y_3d_min, y_3d_max = y_min / fy / z, y_max / fy / z

                x_size = (x_3d_max - x_3d_min) / 2.
                y_size = (y_3d_max - y_3d_min) / 2.
                x_center = (x_3d_max + x_3d_min) / 2.
                y_center = (y_3d_max + y_3d_min) / 2.
                z_center = z
                z_size = torch.ones(1) * 0.1
                rotation = torch.zeros(1)

                z_size = z_size.to(y_size.device)
                rotation = rotation.to(y_size.device)

                prim = torch.stack([x_center, y_center, z_center, x_size, y_size, z_size[0], rotation[0]])
            else:
                prim = torch.zeros(7).to(x.device)

            prims.append(prim)

        prims = torch.stack(prims, dim=0)

        return prims


def rs(x):
    return max(min(x, 255), 0)

class Cube2MaskManual(nn.Module):
    def __init__(self):
        super(Cube2MaskManual, self).__init__()
        # First 4 coordinates are bounding boxes, the last two are the closest and further
        # in depth map
        self.resnet = resnet18(num_classes=7)


    def forward(self, prims):
        segs = []

        fx, fy = 221, 221

        for i, prim in enumerate(prims):
            seg_im = torch.zeros(256, 256).to(prim.device)
            if (prim == 0).all():
                pass
            else:
                prim_numpy = prim.detach().cpu().numpy()
                x_center, y_center, z_center, x_size, y_size, z_size, rot = prim_numpy
                x_min, x_max = x_center - x_size, x_center + x_size
                y_min, y_max = y_center - y_size, y_center + y_size
                # z = (z_center - 1) * 128
                z = z_center
                z_y_min = (z_center - 1) * 128


                x_min, x_max = x_min * z * fx, x_max * z * fx
                y_min, y_max = y_min * z * fy, y_max * z * fy

                x_min, x_max = x_min + 128, x_max + 128
                y_max, y_min = 255 - y_min, 255 - y_max

                x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
                x_min, x_max, y_min, y_max = rs(x_min), rs(x_max), rs(y_min), rs(y_max)

                seg_im[y_min:y_max, x_min:x_max] = 1.

            segs.append(seg_im)

        segs = torch.stack(segs, dim=0)[:, None, :, :]

        return segs


if __name__ == "__main__":
    # Unit test the model
    # model = Mask2Cube().cuda()
    # im = torch.zeros(2, 1, 256, 256).cuda()
    # output = model.forward(im)
    # print(output)

    # Unit test mapping the primitive to mask
    model = Cube2Mask().cuda()
    latent = torch.zeros(2, 7).cuda()
    output = model.forward(latent)
