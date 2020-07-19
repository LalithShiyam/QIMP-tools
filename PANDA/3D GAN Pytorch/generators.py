import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


lrelu = nn.LeakyReLU(0.2)


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
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


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
        elif classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>



# -----  ResnetGenerator -----------

class ResnetBlock(nn.Module):
    def __init__(self, inf, onf):
        """
        Parameters:
            inf: input number of filters
            onf: output number of filters
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(inf, onf)

    def build_conv_block(self, inf, onf):
        conv_block = [nn.Conv3d(inf, onf, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(onf), lrelu]
        conv_block += [nn.Conv3d(inf, onf, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(onf)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class DeUpBlock(nn.Module):
    """Up sample block using torch.nn.ConvTranspose3d"""
    def __init__(self, inf, onf):
        super(DeUpBlock, self).__init__()
        sequence = [nn.ConvTranspose3d(inf, onf, kernel_size=3, stride=1, padding=1), lrelu]
        self.deupblock = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.deupblock(x)
        return out



class UpBlock(nn.Module):
    """Up sample block using torch.nn.Upsample
    """
    def __init__(self, inf, onf):
        super(UpBlock, self).__init__()
        sequence = [nn.Conv3d(inf, onf, kernel_size=3, padding=1), lrelu]
        sequence += [nn.Upsample(scale_factor=2)]
        sequence += [nn.Conv3d(inf, onf, kernel_size=3, padding=1), lrelu]
        self.upblock = nn.Sequential(*sequence)

    def forward(self, x):
        return self.upblock(x)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_residual_blocks=9, upsample_factor=2, deup=True):
        """
        Parameters:
            n_blocks: the number of resnetblocks
            deup: use deconv to upsample
        """
        assert upsample_factor % 2 == 0, "only support even upsample_factor"
        super(ResnetGenerator,self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor
        self.ngf = ngf
        self.deup = deup
        # the first conv-lrelu
        self.conv_blockl_1 = nn.Sequential(nn.Conv3d(input_nc,ngf,kernel_size=3,padding=1),lrelu)
        # residual blocks
        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), ResnetBlock(ngf,ngf))
        # the conv after residual blocks
        self.conv_blockl_2 = nn.Sequential(nn.Conv3d(ngf, ngf, kernel_size=3, padding=1),
                                           nn.BatchNorm3d(ngf))
        # upsample blocks
        for i in range(int(self.upsample_factor/2)):
            if self.deup:
                self.add_module('de_upsample' + str(i+1), DeUpBlock(ngf, ngf))
            else:
                self.add_module('upsample' + str(i+1), UpBlock(ngf, ngf))
        # the last conv
        self.conv3 = nn.Conv3d(ngf, output_nc, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_blockl_1(x)
        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)
        # large skip connection
        x = self.conv_blockl_2(y) + x

        for i in range(int(self.upsample_factor/2)):
            if self.deup:
                x = self.__getattr__('de_upsample' + str(i + 1))(x)
            else:
                x = self.__getattr__('upsample' + str(i+1))(x)

        return self.conv3(x)

# -------------------------------

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=1, output_nc=1, num_downs=6, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=True):
        super(UnetGenerator, self).__init__()

        # currently support only input_nc == output_nc
        assert (input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.BatchNorm3d
        else:
            use_bias = norm_layer == nn.BatchNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.ReLU(True)]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
#
# # -------------------------------

def build_netG(opt):
    if opt.netG == 'resnet':
        generator = ResnetGenerator(n_residual_blocks=9, deup=True)
    elif opt.netG == 'Unet':
        generator = UnetGenerator(ngf=opt.ngf, use_dropout=True, norm_layer=nn.BatchNorm3d)
    else:
        raise NotImplementedError

    init_weights(generator, init_type='normal')

    return generator


if __name__ == '__main__':
    import time
    import torch
    from torch.autograd import Variable
    from torchsummaryX import summary

    from init import Options
    opt = Options().parse()

    torch.cuda.set_device(0)
    generator = build_netG(opt)

    net = generator.cuda().eval()

    data = Variable(torch.randn(4, 1, 128, 128, 64)).cuda()

    out = net(data)

    summary(net,data)
    print("out size: {}".format(out.size()))









