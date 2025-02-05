import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import ParametricLIFNode
from spikingjelly.clock_driven import layer

def conv3x3(in_channels, out_channels):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ),
        ParametricLIFNode(init_tau=2.0, detach_reset=True, step_mode='m')
    )

def ms_conv3x3(in_channels, out_channels):
    return nn.Sequential(
        ParametricLIFNode(init_tau=2.0, detach_reset=True, step_mode='m'),
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    )

def conv1x1(in_channels, out_channels):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ),
        ParametricLIFNode(init_tau=2.0, detach_reset=True, step_mode='m')
    )

def ms_conv1x1(in_channels, out_channels):
    return nn.Sequential(
        ParametricLIFNode(init_tau=2.0, detach_reset=True, step_mode='m'),
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    )

class SEWBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, connect_f=None):
        super(SEWBlock, self).__init__()
        self.connect_f = connect_f
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels),
            conv3x3(mid_channels, in_channels),
        )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        if self.connect_f == 'ADD':
            out += x
        elif self.connect_f == 'AND':
            out *= x
        elif self.connect_f == 'IAND':
            out = x * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels),

            layer.SeqToANNContainer(
                nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(in_channels),
            ),
        )
        self.sn = ParametricLIFNode(init_tau=2.0, detach_reset=True, step_mode='m')

    def forward(self, x: torch.Tensor):
        return self.sn(x + self.conv(x))


class MSBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(MSBlock, self).__init__()
        self.sn = ParametricLIFNode(init_tau=2.0, detach_reset=True, step_mode='m')
        self.conv = nn.Sequential(
            ms_conv3x3(in_channels, mid_channels),
            ms_conv3x3(mid_channels, in_channels),
        )
    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        out += x
        return out

class ResNetN(nn.Module):
    def __init__(self, layer_list, num_classes, connect_f=None, model=str):
        super(ResNetN, self).__init__()
        in_channels = 2
        conv = []
        self.model = model

        for cfg_dict in layer_list:
            channels = cfg_dict['channels']

            if 'mid_channels' in cfg_dict:
                mid_channels = cfg_dict['mid_channels']
            else:
                mid_channels = channels

            if in_channels != channels:
                if self.model in ['sew', 'basic']:
                    if cfg_dict['up_kernel_size'] == 3:
                        conv.append(conv3x3(in_channels, channels))
                    elif cfg_dict['up_kernel_size'] == 1:
                        conv.append(conv1x1(in_channels, channels))
                    else:
                        raise NotImplementedError
                elif self.model == 'ms':
                    if cfg_dict['up_kernel_size'] == 3:
                        conv.append(ms_conv3x3(in_channels, channels))
                    elif cfg_dict['up_kernel_size'] == 1:
                        conv.append(ms_conv1x1(in_channels, channels))
                    else:
                        raise NotImplementedError

            in_channels = channels


            if 'num_blocks' in cfg_dict:
                num_blocks = cfg_dict['num_blocks']
                if cfg_dict['block_type'] == 'sew':
                    for _ in range(num_blocks):
                        conv.append(SEWBlock(in_channels, mid_channels, connect_f))
                elif cfg_dict['block_type'] == 'basic':
                    for _ in range(num_blocks):
                        conv.append(BasicBlock(in_channels, mid_channels))
                elif cfg_dict['block_type'] == 'ms':
                    for _ in range(num_blocks):
                        conv.append(MSBlock(in_channels, mid_channels))        
                else:
                    raise NotImplementedError

            if 'k_pool' in cfg_dict:
                k_pool = cfg_dict['k_pool']
                conv.append(layer.SeqToANNContainer(nn.MaxPool2d(k_pool, k_pool)))

        conv.append(nn.Flatten(2))

        self.conv = nn.Sequential(*conv)
        self.lastLIF = ParametricLIFNode(init_tau=2.0, detach_reset=True, step_mode='m')

        with torch.no_grad():
            x = torch.zeros([1, 1, 128, 128])
            for m in self.conv.modules():
                if isinstance(m, nn.MaxPool2d):
                    x = m(x)
            out_features = x.numel() * in_channels  #SEWResNet=32 x텐서의 총 원소 수를 계산
            print(x.shape)
            print(in_channels, out_features)
        self.out = nn.Linear(out_features, num_classes, bias=True)

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        print(x.shape)
        x = self.conv(x)
        # if self.model in ['sew', 'basic']:
        #     x = self.conv(x)
        # elif self.model == 'ms':
        #     x = self.conv(x)
        #     #x = F.adaptive_avg_pool3d(x, (None, 1, 1))
        #     #x = self.lastLIF(x)
        return self.out(x.mean(0))

def SEWResNet(connect_f):
    layer_list = [
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
    ]
    num_classes = 11
    model = 'sew'
    return ResNetN(layer_list, num_classes, connect_f, model)

def SpikingResNet(*args, **kwargs):
    layer_list = [
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
    ]
    num_classes = 11
    model = 'basic'
    return ResNetN(layer_list, num_classes, model)

def MSResNet(*args, **kwargs):
    layer_list = [
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'ms', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'ms', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'ms', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'ms', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'ms', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'ms', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'ms', 'k_pool': 2},
    ]
    num_classes = 11
    model = 'ms'
    return ResNetN(layer_list, num_classes, model)