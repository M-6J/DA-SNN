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

def ms_input_conv3x3(in_channels, out_channels):
        return nn.Sequential(
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

def ms_input_conv1x1(in_channels, out_channels):
        return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1, stride=1, bias=False),
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
    def __init__(self, layer_list, num_classes, connect_f=None, ms=True):
        super(ResNetN, self).__init__()
        in_channels = 2
        conv = []
        self.use_ms = ms
        print(self.use_ms)

        for cfg_dict in layer_list:
            channels = cfg_dict['channels']

            if 'mid_channels' in cfg_dict:
                mid_channels = cfg_dict['mid_channels']
            else:
                mid_channels = channels

            if in_channels != channels: # 2 != 32
                if self.use_ms is True:
                    if cfg_dict['up_kernel_size'] == 3:
                        conv.append(ms_input_conv3x3(in_channels, channels))
                    elif cfg_dict['up_kernel_size'] == 1:
                        conv.append(ms_input_conv1x1(in_channels, channels))
                    else:
                        raise NotImplementedError
                else:
                    if cfg_dict['up_kernel_size'] == 3:
                        conv.append(conv3x3(in_channels, channels))
                    elif cfg_dict['up_kernel_size'] == 1:
                        conv.append(conv1x1(in_channels, channels))
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

        #conv.append(nn.Flatten(2))

        self.conv = nn.Sequential(*conv)
        self.lastLIF = ParametricLIFNode(init_tau=2.0, detach_reset=True, step_mode='m')

        with torch.no_grad():
            x = torch.zeros([1, 1, 128, 128])
            for m in self.conv.modules():
                if isinstance(m, nn.MaxPool2d):
                    x = m(x)
            out_features = x.numel() * in_channels  #SEWResNet=32 x텐서의 총 원소 수를 계산
            print(in_channels, out_features)
        self.out = nn.Linear(out_features, num_classes, bias=True)

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        if self.use_ms is True:
            print(x.shape) #torch.Size([5, 16, 2, 128, 128])
            x = self.conv(x)
            print(x.shape) # torch.Size([5, 16, 32, 1, 1])
            x = self.lastLIF(x)
            print(x.shape) #torch.Size([5, 16, 32, 1, 1])
            x = x.flatten(2)
            print(x.shape) #torch.Size([5, 16, 32])
        else:
            x = self.conv(x)
            x = x.flatten(2)
        return self.out(x.mean(0))

def SEWResNet(connect_f):
    layer_list = [
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2}, #64x64    #image size
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2}, #32x32
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2}, #16x16
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2}, #8x8
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2}, #4x4
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2}, #2x2
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2}, #1x1
    ]
    num_classes = 11
    return ResNetN(layer_list, num_classes, connect_f, ms=False)

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
    return ResNetN(layer_list, num_classes, ms=False)

def MSResNet(*args, **kwargs):
    layer_list = [ 
        {'channels': 32, 'up_kernel_size': 3, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'ms', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'ms', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'ms', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'ms', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'ms', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'ms', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'ms', 'k_pool': 2},
    ]
    num_classes = 11
    model = 'ms'
    return ResNetN(layer_list, num_classes, ms=True)



#https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based/classify_dvsg.html
# #from copy import deepcopy
# #from spikingjelly.activation_based import layer
# class DVSGestureNet(nn.Module):
#     def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
#         super().__init__()

#         conv = []
#         for i in range(5):
#             if conv.__len__() == 0:
#                 in_channels = 2
#             else:
#                 in_channels = channels

#             conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
#             conv.append(layer.BatchNorm2d(channels))
#             conv.append(spiking_neuron(**deepcopy(kwargs)))
#             conv.append(layer.MaxPool2d(2, 2))


#         self.conv_fc = nn.Sequential(
#             *conv,

#             layer.Flatten(),
#             layer.Dropout(0.5),
#             layer.Linear(channels * 4 * 4, 512),
#             spiking_neuron(**deepcopy(kwargs)),

#             layer.Dropout(0.5),
#             layer.Linear(512, 110),
#             spiking_neuron(**deepcopy(kwargs)),

#             layer.VotingLayer(10)
#         )

#     def forward(self, x: torch.Tensor):
#         return self.conv_fc(x)