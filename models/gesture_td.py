import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import ParametricLIFNode
from spikingjelly.clock_driven import layer
from models.DTA import DTA
from models.layers import *

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
class MSBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=32, dilation=1, norm_layer=None):
        super(MSBlock, self).__init__()
        if norm_layer is None:
            norm_layer = tdBatchNorm
        if groups != 1 or base_width != 32:
            raise ValueError('BasicBlock only supports groups=1 and base_width=32')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.conv2_s = tdLayer(self.conv2, self.bn2)
        self.spike = LIFSpike()
        
    def forward(self, x):
        identity = x
        out = self.spike(x)
        out = self.conv1_s(out)
        out = self.spike(out)
        out = self.conv2_s(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        
        return out

class ResNetN(nn.Module):
    def __init__(self, block, layers, num_classes=11, time_step=5, DTA_ON=True, 
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetN, self).__init__()
        in_channels = 2
        conv = []
        self.use_dta = DTA_ON
        self.inplanes = 32
        norm_layer = tdBatchNorm
        self._norm_layer = norm_layer

        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        self.groups = 1
        self.base_width = 32

        self.T = time_step

        if self.use_dta==True:
            self.encoding = DTA(T=self.T , out_channels = 32)
        else: 
            self.encoding = None

        self.input_conv = tdLayer(nn.Conv2d(2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), 
                                    norm_layer(self.inplanes))
        
        self.LIF = LIFSpike()
        self.conv = self._make_layer(block, 32, layers[0])
        #self.MP = tdLayer(nn.MaxPool2d((1, 1)))

        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))

        self.fc = tdLayer(nn.Linear(32, 11))

        ###fc 추가해야함

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tdLayer(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        layers.append(tdLayer(nn.MaxPool2d((1, 1))))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
            layers.append(tdLayer(nn.MaxPool2d((1, 1))))

        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *] torch.Size([5, 16, 2, 128, 128])
        x = self.input_conv(x)
        img = x
        x = self.LIF(x)
        x = self.encoding(img,x)
        x = self.conv(x)
        x = self.LIF(x)
        print(x.shape)
        x = self.avgpool(x)
        print(x.shape)
        x = torch.flatten(x, 2)
        print(x.shape)
        x = self.fc(x)
        return x


def ms_resnet(block, layers, **kwargs):
    
    model = ResNetN(block, layers, time_step=5,**kwargs)
    return model

def MSResNet(**kwargs):
    return ms_resnet(MSBlock, [7],
                   **kwargs)