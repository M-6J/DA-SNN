import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import ParametricLIFNode
from spikingjelly.clock_driven import layer
from models.DTA import DTA

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
    def __init__(self, layer_list, num_classes, DTA_ON=True, ms=True):
        super(ResNetN, self).__init__()
        in_channels = 2
        conv = []
        self.use_ms = ms
        self.use_dta = DTA_ON
        print('ms = ', self.use_ms)
        print('DTA = ', self.use_dta)
        self.T = 5

        if self.use_dta==True:
            self.encoding = DTA(T=self.T , out_channels = 32)
        else: 
            self.encoding = None

        self.input_conv = ms_input_conv3x3(in_channels, 32)

        self.LIF = nn.Sequential(ParametricLIFNode(init_tau=2.0, detach_reset=True, step_mode='m'))

        for cfg_dict in layer_list:
            channels = cfg_dict['channels']

            if 'mid_channels' in cfg_dict:
                mid_channels = cfg_dict['mid_channels']
            else:
                mid_channels = channels

            in_channels = channels


            if 'num_blocks' in cfg_dict:
                num_blocks = cfg_dict['num_blocks']
                if cfg_dict['block_type'] == 'ms':
                    for _ in range(num_blocks):
                        conv.append(MSBlock(in_channels, mid_channels))        
                else:
                    raise NotImplementedError

            if 'k_pool' in cfg_dict:
                k_pool = cfg_dict['k_pool']
                conv.append(layer.SeqToANNContainer(nn.MaxPool2d(k_pool, k_pool)))

         #conv.append(nn.Flatten(2))

        self.conv = nn.Sequential(*conv)

        with torch.no_grad():
            x = torch.zeros([1, 1, 128, 128])
            for m in self.conv.modules():
                if isinstance(m, nn.MaxPool2d):
                    x = m(x)
            out_features = x.numel() * in_channels  #SEWResNet=32 x텐서의 총 원소 수를 계산
            print(in_channels, out_features)
        self.out = nn.Linear(out_features, num_classes, bias=True)

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *] torch.Size([5, 16, 2, 128, 128])
        if self.use_ms is True:
            x = self.input_conv(x)
            #print(x.shape) # torch.Size([5, 16, 32, 128, 128])
            img = x
            #print(x.shape) # torch.Size([5, 16, 32, 128, 128])
            x = self.encoding(img,x) #attention
            #print(x.shape) # torch.Size([5, 16, 32, 128, 128])
            x = self.conv(x)
            #print(x.shape) # torch.Size([5, 16, 32, 1, 1])
            x = self.LIF(x)
            #print(x.shape) #torch.Size([5, 16, 32, 1, 1])
            x = x.flatten(2)
            #print(x.shape) #torch.Size([5, 16, 32])
        else:
            x = self.conv(x)
            x = x.flatten(2)
        return self.out(x.mean(0))


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
    return ResNetN(layer_list, num_classes, DTA_ON=True, ms=True)

