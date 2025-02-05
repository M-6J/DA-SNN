from models.layers import *
from models.DTA import DTA

class MS_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=11, time_step=5, DTA_ON=True, dvs=None):
        super(MS_ResNet, self).__init__()
        
        self.dvs = dvs     

        self.T = time_step # time-step
        norm_layer = tdBatchNorm
        self._norm_layer = norm_layer
        self.inplanes = 32
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        self.groups = 1
        self.base_width = 32
        if self.dvs is True: 
            self.input_conv = tdLayer(nn.Conv2d(2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), 
                                    norm_layer(self.inplanes))
        else:
            self.input_conv = tdLayer(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), 
                                  norm_layer(self.inplanes))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))

        self.fc = tdLayer(nn.Linear(256 * block.expansion, num_classes))
        self.LIF = LIFSpike()
        
        if DTA_ON==True:
            self.encoding = DTA(T=self.T, out_channels = 32)
        else: 
            self.encoding = None
            
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
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.input_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.LIF(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x) 


class SEW_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=11, time_step=5, DTA_ON=True, dvs=None):
        super(SEW_ResNet, self).__init__()
        
        self.dvs = dvs     

        self.T = time_step # time-step
        norm_layer = tdBatchNorm
        self._norm_layer = norm_layer
        self.inplanes = 32
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        self.groups = 1
        self.base_width = 32
        if self.dvs is True: 
            self.input_conv = tdLayer(nn.Conv2d(2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), 
                                    norm_layer(self.inplanes))
        else:
            self.input_conv = tdLayer(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), 
                                  norm_layer(self.inplanes))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))

        self.fc = tdLayer(nn.Linear(256 * block.expansion, num_classes))
        self.LIF = LIFSpike()
        
        if DTA_ON==True:
            self.encoding = DTA(T=self.T, out_channels = 32)
        else: 
            self.encoding = None
            
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
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.input_conv(x)
        x = self.LIF(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x) 



def ms_resnet(block, layers, **kwargs):
    model = MS_ResNet(block, layers, **kwargs)
    return model


def dta_msresnet18(**kwargs):
    return ms_resnet(BasicBlock_MS, [3, 3, 2],
                   **kwargs)

def sew_resnet(block, layers, **kwargs):
    model = SEW_ResNet(block, layers, **kwargs)
    return model


def dta_sewresnet18(**kwargs):
    return sew_resnet(BasicBlock_SEW, [3, 3, 2],
                   **kwargs)