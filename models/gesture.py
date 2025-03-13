import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import ParametricLIFNode
from spikingjelly.clock_driven import layer
from models.DTA import DTA
from models.GAU import TA,SCA

class GAC(nn.Module):
    def __init__(self,T,out_channels):
        super().__init__()
        self.TA = TA(T = T)
        self.SCA = SCA(in_planes= out_channels,kerenel_size=4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq, spikes):
        # x_seq B T inplanes H W
        # spikes B T inplanes H W
        x_seq = x_seq.permute(1, 0, 2, 3, 4)
        spikes = spikes.permute(1, 0, 2, 3, 4)

        TA = self.TA(x_seq)
        SCA = self.SCA(x_seq)
        out = self.sigmoid(TA * SCA)
        y_seq = out * spikes
        return y_seq.permute(1, 0, 2, 3, 4)

def conv3x3(in_channels, out_channels):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ),
        ParametricLIFNode(init_tau=2.0, detach_reset=True, step_mode='m')
    )

def conv1x1(in_channels, out_channels):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
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
        out += x
        return out

class ResNetN(nn.Module):
    def __init__(self, layer_list, num_classes, time_step=20, DTA_ON=True, ms=True):
        super(ResNetN, self).__init__()
        in_channels = 2
        conv = []
        self.use_ms = ms
        self.use_dta = DTA_ON
        print('ms = ', self.use_ms)
        print('DTA = ', self.use_dta)
        self.T = time_step

        if self.use_dta==True:
            #self.encoding = DTA(T=self.T , out_channels = 32)
            self.encoding = GAC(T=self.T , out_channels = 32)
        else: 
            self.encoding = None

        self.input_conv = ms_input_conv3x3(in_channels, 32)
        print(in_channels)

        self.LIF = ParametricLIFNode(init_tau=2.0, detach_reset=True, step_mode='m')
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.mp2 = layer.SeqToANNContainer(nn.MaxPool2d(2, 2))
        self.maxpool = layer.SeqToANNContainer(nn.AdaptiveMaxPool2d((1, 1)))

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
                elif cfg_dict['block_type'] == 'sew':
                    for _ in range(num_blocks):
                        conv.append(SEWBlock(in_channels, mid_channels))         
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
            img = x
            x = self.LIF(x)
            x = self.encoding(img,x) #attention
            x = mp2(x)
            x = self.conv(x)
            #x = self.encoding(x)
            #x = self.LIF(x)
            #x = self.avgpool(x)
            x = self.maxpool(x)
            x = x.flatten(2)
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
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'ms'},#, 'k_pool': 2},
    ]
    num_classes = 11
    model = 'ms'
    time_step = 5
    return ResNetN(layer_list, num_classes, time_step, DTA_ON=True, ms=True)

def SEWResNet(*args, **kwargs):
    layer_list = [
        {'channels': 32, 'up_kernel_size': 3, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},  # 64x64 
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},  # 32x32 
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},  # 16x16 
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2}, # 8x8 
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2}, # 4x4 
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2}, # 2x2 
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew'},#, 'k_pool': 2}, # 1x1 
    ]
    num_classes = 11
    time_step = 5
    return ResNetN(layer_list, num_classes, time_step, DTA_ON=True, ms=True)

