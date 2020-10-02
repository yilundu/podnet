import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


##############################################################################
# Classes
##############################################################################
class Flatten(nn.Module):

    def forward(self, x):
        return x.flatten(start_dim=1)


class ComponentVAE(nn.Module):

    def __init__(self, input_nc, z_dim=16, full_res=False, frames=1):
        super().__init__()
        self._input_nc = input_nc
        self._z_dim = z_dim
        self.frames = frames

        ngf = z_dim
        self.conv1 = nn.Conv2d(input_nc * frames + frames, ngf, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(ngf, affine=True)
        # self.bn1 = nn.GroupNorm(32, ngf, affine=True)
        self.bn1 = nn.InstanceNorm2d(ngf, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Downsample
        self.res1 = BasicBlock(ngf, 2*ngf, stride=2, remap=nn.Conv2d(ngf, 2*ngf, kernel_size=1, stride=2))
        self.res2 = BasicBlock(2*ngf, 4*ngf, stride=2, remap=nn.Conv2d(2*ngf, 4*ngf, kernel_size=1, stride=2))
        self.res3 = BasicBlock(4*ngf, 8*ngf, stride=2, remap=nn.Conv2d(4*ngf, 8*ngf, kernel_size=1, stride=2))

        self.fc1 = nn.Linear(8*ngf, 256)
        self.fc2 = nn.Linear(256, 2 * z_dim)


        self.decode = nn.Conv2d(z_dim + 2, 8*ngf, kernel_size=3, padding=1)
        self.res1_upsample = BasicBlock(8*ngf, 4*ngf, remap=conv3x3_upsample(8*ngf, 4*ngf), upsample=True)
        self.res2_upsample = BasicBlock(4*ngf, 2*ngf, remap=conv3x3_upsample(4*ngf, 2*ngf), upsample=True)
        self.res3_upsample = BasicBlock(2*ngf, ngf, remap=conv3x3_upsample(2*ngf, ngf), upsample=True)
        self.res4_upsample = BasicBlock(ngf, ngf, remap=conv3x3_upsample(ngf, ngf), upsample=True)
        self.res5_upsample = BasicBlock(ngf, ngf, remap=conv3x3_upsample(ngf, ngf), upsample=True)
        self.output = nn.Conv2d(ngf, input_nc * frames + frames, 1)
        self._bg_logvar = 2 * torch.tensor(0.07).log()
        self._fg_logvar = 2 * torch.tensor(0.11).log()

        # full_res = False # full res: 128x128, low res: 64x64
        # h_dim = 4096 if full_res else 1024
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(input_nc * frames + frames, 32, 3, stride=2, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 32, 3, stride=2, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, 3, stride=2, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 64, 3, stride=2, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 64, 3, stride=2, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 64, 3, stride=2, padding=1),
        #     nn.ReLU(True),
        #     Flatten(),
        #     nn.Linear(h_dim, 256),
        #     nn.ReLU(True),
        #     nn.Linear(256, 2 * z_dim)
        # )
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(z_dim + 2, 32, 3),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 32, 3),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 32, 3),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 32, 3),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, input_nc * frames + frames, 1),
        # )
        # self._bg_logvar = 2 * torch.tensor(0.09).log()
        # self._fg_logvar = 2 * torch.tensor(0.11).log()
        # self.frames = frames

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    @staticmethod
    def spatial_broadcast(z, h, w):
        # Batch size
        n = z.shape[0]
        # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
        z_b = z.view((n, -1, 1, 1)).expand(-1, -1, h, w)
        # Coordinate axes:
        x = torch.linspace(-1, 1, w, device=z.device)
        y = torch.linspace(-1, 1, h, device=z.device)
        y_b, x_b = torch.meshgrid(y, x)
        # Expand from (h, w) -> (n, 1, h, w)
        x_b = x_b.expand(n, 1, -1, -1)
        y_b = y_b.expand(n, 1, -1, -1)
        # Concatenate along the channel dimension: final shape = (n, z_dim + 2, h, w)
        z_sb = torch.cat((z_b, x_b, y_b), dim=1)
        return z_sb

    def forward(self, x, log_m_k, background=False):
        """
        :param x: Input image
        :param log_m_k: Attention mask logits
        :return: x_k and reconstructed mask logits
        """
        inp = torch.cat([x, log_m_k], dim=1)
        inp = self.conv1(inp)
        inp = self.relu(self.bn1(inp))
        inp = self.maxpool(inp)
        inp = self.res1(inp)
        inp = self.res2(inp)
        inp = self.res3(inp)
        inp = inp.mean(dim=[2, 3])
        params = self.fc2(F.relu(self.fc1(inp)))

        z_mu = params[:, :self._z_dim]
        z_logvar = params[:, self._z_dim:]
        z = self.reparameterize(z_mu, z_logvar)

        z_sb = self.spatial_broadcast(z, 8, 8)
        z_sb = F.relu(self.decode(z_sb))
        z_sb = self.res1_upsample(z_sb)
        z_sb = self.res2_upsample(z_sb)
        z_sb = self.res3_upsample(z_sb)
        z_sb = self.res4_upsample(z_sb)
        z_sb = self.res5_upsample(z_sb)
        output = self.output(z_sb)
        x_mu = output[:, :self._input_nc * self.frames]
        x_logvar = self._bg_logvar if background else self._fg_logvar
        m_logits = output[:, self._input_nc * self.frames:]

        # params = self.encoder(torch.cat((x, log_m_k), dim=1))
        # z_mu = params[:, :self._z_dim]
        # z_logvar = params[:, self._z_dim:]
        # z = self.reparameterize(z_mu, z_logvar)

        # # "The height and width of the input to this CNN were both 8 larger than the target output (i.e. image) size
        # #  to arrive at the target size (i.e. accommodating for the lack of padding)."
        # h, w = x.shape[-2:]
        # z_sb = self.spatial_broadcast(z, h + 8, w + 8)

        # output = self.decoder(z_sb)
        # x_mu = output[:, :self._input_nc * self.frames]
        # x_logvar = self._bg_logvar if background else self._fg_logvar
        # m_logits = output[:, self._input_nc * self.frames:]

        return m_logits, x_mu, x_logvar, z_mu, z_logvar


class ComponentVAEOld(nn.Module):

    def __init__(self, input_nc, z_dim=16, full_res=False, frames=1):
        super().__init__()
        self._input_nc = input_nc
        self._z_dim = z_dim
        self.frames = frames

        full_res = False # full res: 128x128, low res: 64x64
        h_dim = 4096 if full_res else 1024
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc * frames + frames, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(h_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 2 * z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim + 2, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, input_nc * frames + frames, 1),
        )
        self._bg_logvar = 2 * torch.tensor(0.09).log()
        self._fg_logvar = 2 * torch.tensor(0.11).log()
        self.frames = frames

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    @staticmethod
    def spatial_broadcast(z, h, w):
        # Batch size
        n = z.shape[0]
        # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
        z_b = z.view((n, -1, 1, 1)).expand(-1, -1, h, w)
        # Coordinate axes:
        x = torch.linspace(-1, 1, w, device=z.device)
        y = torch.linspace(-1, 1, h, device=z.device)
        y_b, x_b = torch.meshgrid(y, x)
        # Expand from (h, w) -> (n, 1, h, w)
        x_b = x_b.expand(n, 1, -1, -1)
        y_b = y_b.expand(n, 1, -1, -1)
        # Concatenate along the channel dimension: final shape = (n, z_dim + 2, h, w)
        z_sb = torch.cat((z_b, x_b, y_b), dim=1)
        return z_sb

    def forward(self, x, log_m_k, background=False):
        """
        :param x: Input image
        :param log_m_k: Attention mask logits
        :return: x_k and reconstructed mask logits
        """
        params = self.encoder(torch.cat((x, log_m_k), dim=1))
        z_mu = params[:, :self._z_dim]
        z_logvar = params[:, self._z_dim:]
        z = self.reparameterize(z_mu, z_logvar)

        # "The height and width of the input to this CNN were both 8 larger than the target output (i.e. image) size
        #  to arrive at the target size (i.e. accommodating for the lack of padding)."
        h, w = x.shape[-2:]
        z_sb = self.spatial_broadcast(z, h + 8, w + 8)

        output = self.decoder(z_sb)
        x_mu = output[:, :self._input_nc * self.frames]
        x_logvar = self._bg_logvar if background else self._fg_logvar
        m_logits = output[:, self._input_nc * self.frames:]

        return m_logits, x_mu, x_logvar, z_mu, z_logvar


class AttentionBlock(nn.Module):

    def __init__(self, input_nc, output_nc, resize=True):
        super().__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(output_nc, affine=True)
        self._resize = resize

    def forward(self, *inputs):
        downsampling = len(inputs) == 1
        x = inputs[0] if downsampling else torch.cat(inputs, dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = skip = F.relu(x)
        if self._resize:
            x = F.interpolate(skip, scale_factor=0.5 if downsampling else 2., mode='nearest')
        return (x, skip) if downsampling else x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_upsample(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, remap=None, upsample=False):
        super(BasicBlock, self).__init__()

        if upsample:
            self.conv1 = conv3x3_upsample(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)

        # self.bn1 = nn.GroupNorm(32, planes)
        self.bn1 = nn.InstanceNorm2d(planes, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = nn.GroupNorm(32, planes)
        self.bn2 = nn.InstanceNorm2d(planes, affine=True)
        self.remap = remap
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.remap is not None:
            residual = self.remap(x)

        out += residual
        out = self.relu(out)

        return out


class Attention(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(input_nc + output_nc, ngf, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Downsample
        self.res1 = BasicBlock(ngf, 2*ngf, stride=2, remap=nn.Conv2d(ngf, 2*ngf, kernel_size=1, stride=2))
        self.res2 = BasicBlock(2*ngf, 4*ngf, stride=2, remap=nn.Conv2d(2*ngf, 4*ngf, kernel_size=1, stride=2))
        self.res3 = BasicBlock(4*ngf, 8*ngf, stride=2, remap=nn.Conv2d(4*ngf, 8*ngf, kernel_size=1, stride=2))

        # Upsample
        self.res1_upsample = BasicBlock(8*ngf, 4*ngf, remap=conv3x3_upsample(8*ngf, 4*ngf), upsample=True)
        self.res2_upsample = BasicBlock(4*ngf, 2*ngf, remap=conv3x3_upsample(4*ngf, 2*ngf), upsample=True)
        self.res3_upsample = BasicBlock(2*ngf, ngf, remap=conv3x3_upsample(2*ngf, ngf), upsample=True)
        self.res4_upsample = BasicBlock(ngf, ngf, remap=conv3x3_upsample(ngf, ngf), upsample=True)
        self.res5_upsample = BasicBlock(ngf, ngf, remap=conv3x3_upsample(ngf, ngf), upsample=True)
        self.output = nn.Conv2d(ngf, output_nc, 1)

        # Old attention network
        # self.downblock1 = AttentionBlock(input_nc + output_nc, ngf)
        # self.downblock2 = AttentionBlock(ngf, ngf * 2)
        # self.downblock3 = AttentionBlock(ngf * 2, ngf * 4)
        # self.downblock4 = AttentionBlock(ngf * 4, ngf * 4)
        # self.downblock5 = AttentionBlock(ngf * 4, ngf * 4)
        # self.downblock6 = AttentionBlock(ngf * 4, ngf * 4)
        # self.downblock7 = AttentionBlock(ngf * 4, ngf * 4)
        # # no resizing occurs in the last block of each path
        # # self.downblock6 = AttentionBlock(ngf * 8, ngf * 8, resize=False)

        # self.mlp = nn.Sequential(
        #     nn.Linear(4 * 4 * ngf * 4, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 4 * 4 * ngf * 4),
        #     nn.ReLU(),
        # )

        # # self.upblock1 = AttentionBlock(2 * ngf * 8, ngf * 8)
        # self.upblock2 = AttentionBlock(2 * ngf * 4, ngf * 4)
        # self.upblock3 = AttentionBlock(2 * ngf * 4, ngf * 4)
        # self.upblock4 = AttentionBlock(2 * ngf * 4, ngf * 4)

        # self.upblock5 = AttentionBlock(2 * ngf * 4, ngf * 4)
        # self.upblock6 = AttentionBlock(2 * ngf * 4, ngf * 2)
        # self.upblock7 = AttentionBlock(2 * ngf * 2, ngf)
        # # no resizing occurs in the last block of each path
        # self.upblock8 = AttentionBlock(2 * ngf, ngf, resize=False)


    def forward(self, x, log_s_k):
        x = torch.cat([x, log_s_k], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res1_upsample(x)
        x = self.res2_upsample(x)
        x = self.res3_upsample(x)
        x = self.res4_upsample(x)
        x = self.res5_upsample(x)
        x = self.output(x)
        x = F.logsigmoid(x)

        # Old code
        # Downsampling blocks
        # x, skip1 = self.downblock1(torch.cat((x, log_s_k), dim=1))
        # x, skip2 = self.downblock2(x)
        # x, skip3 = self.downblock3(x)
        # x, skip4 = self.downblock4(x)
        # x, skip5 = self.downblock5(x)
        # x, skip6 = self.downblock6(x)
        # x, skip7 = self.downblock7(x)
        # # The input to the MLP is the last skip tensor collected from the downsampling path (after flattening)
        # # _, skip6 = self.downblock6(x)
        # # Flatten
        # x = skip7.flatten(start_dim=1)
        # x = self.mlp(x)
        # # Reshape to match shape of last skip tensor
        # x = x.view(skip7.shape)
        # # Upsampling blocks
        # # x = self.upblock1(x, skip6)
        # x = self.upblock2(x, skip7)
        # x = self.upblock3(x, skip6)
        # x = self.upblock4(x, skip5)
        # x = self.upblock5(x, skip4)
        # x = self.upblock6(x, skip3)
        # x = self.upblock7(x, skip2)
        # x = self.upblock8(x, skip1)
        # Output layer
        # x = self.output(x)
        # x = F.logsigmoid(x)
        return x


class AttentionOld(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(AttentionOld, self).__init__()
        # Old attention network
        self.downblock1 = AttentionBlock(input_nc + output_nc, ngf)
        self.downblock2 = AttentionBlock(ngf, ngf * 2)
        self.downblock3 = AttentionBlock(ngf * 2, ngf * 4)
        self.downblock4 = AttentionBlock(ngf * 4, ngf * 4)
        self.downblock5 = AttentionBlock(ngf * 4, ngf * 4)
        self.downblock6 = AttentionBlock(ngf * 4, ngf * 4)
        self.downblock7 = AttentionBlock(ngf * 4, ngf * 4)
        # no resizing occurs in the last block of each path
        # self.downblock6 = AttentionBlock(ngf * 8, ngf * 8, resize=False)

        self.mlp = nn.Sequential(
            nn.Linear(4 * 4 * ngf * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4 * 4 * ngf * 4),
            nn.ReLU(),
        )

        # self.upblock1 = AttentionBlock(2 * ngf * 8, ngf * 8)
        self.upblock2 = AttentionBlock(2 * ngf * 4, ngf * 4)
        self.upblock3 = AttentionBlock(2 * ngf * 4, ngf * 4)
        self.upblock4 = AttentionBlock(2 * ngf * 4, ngf * 4)

        self.upblock5 = AttentionBlock(2 * ngf * 4, ngf * 4)
        self.upblock6 = AttentionBlock(2 * ngf * 4, ngf * 2)
        self.upblock7 = AttentionBlock(2 * ngf * 2, ngf)
        # no resizing occurs in the last block of each path
        self.upblock8 = AttentionBlock(2 * ngf, ngf, resize=False)
        self.output = nn.Conv2d(ngf, output_nc, 1)


    def forward(self, x, log_s_k):

        # Old code
        # Downsampling blocks
        x, skip1 = self.downblock1(torch.cat((x, log_s_k), dim=1))
        x, skip2 = self.downblock2(x)
        x, skip3 = self.downblock3(x)
        x, skip4 = self.downblock4(x)
        x, skip5 = self.downblock5(x)
        x, skip6 = self.downblock6(x)
        x, skip7 = self.downblock7(x)
        # The input to the MLP is the last skip tensor collected from the downsampling path (after flattening)
        # _, skip6 = self.downblock6(x)
        # Flatten
        x = skip7.flatten(start_dim=1)
        x = self.mlp(x)
        # Reshape to match shape of last skip tensor
        x = x.view(skip7.shape)
        # Upsampling blocks
        # x = self.upblock1(x, skip6)
        x = self.upblock2(x, skip7)
        x = self.upblock3(x, skip6)
        x = self.upblock4(x, skip5)
        x = self.upblock5(x, skip4)
        x = self.upblock6(x, skip3)
        x = self.upblock7(x, skip2)
        x = self.upblock8(x, skip1)
        # Output layer
        x = self.output(x)
        x = F.logsigmoid(x)
        return x
