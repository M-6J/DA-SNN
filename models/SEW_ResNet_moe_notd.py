from models.layers import *
from models.DTA import DTA
from spikingjelly.activation_based.neuron import ParametricLIFNode

#for conv3d 채널기반
class MoEGateNetwork(nn.Module):
    def __init__(self, in_channels, num_experts):
        super().__init__()
        self.conv1 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(128)  # 🔥 Conv3D에 맞게 변경
        self.conv2 = nn.Conv3d(128, num_experts, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(num_experts)  # 🔥 Conv3D에 맞게 변경
        self.ReLU = nn.ReLU()
        self.Softmax = nn.Softmax(dim=1)  

    def forward(self, x):
        # 기존 x.shape: (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # 🔥 Conv3D는 (B, C, T, H, W) 입력 형식이므로 변환
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # 🔥 Global Average Pooling (시간 + 공간 차원 압축)
        x = torch.mean(x, dim=(3, 4))  # (B, num_experts, T) → 공간 차원(H, W) 평균
        x = x.mean(dim=2)  # (B, num_experts) → 시간 차원(T) 평균

        return self.Softmax(x)  # 🔥 Expert 선택 확률 출력


# class MoEGateNetwork(nn.Module):
#     def __init__(self, in_channels, num_experts):
#         super().__init__()
#         # 여기서는 Conv 기반으로 설계해서, shared feature의 spatial 정보를 유지
#         norm_layer = tdBatchNorm
#         self.conv1 = tdLayer(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), norm_layer(32))
#         self.conv2 = tdLayer(nn.Conv2d(32, num_experts, kernel_size=3, padding=1), norm_layer(4))
#         self.ReLU = tdLayer(nn.ReLU())
#         self.Softmax = nn.Softmax(dim=1)
#         

#     def forward(self, x):
#        #print(x.shape) torch.Size([16, 5, 32, 1, 1])
#         x = self.conv1(x)
#         x = self.ReLU(x)
#         x = self.conv2(x)
#         #print(x.shape) #torch.Size([16, 5, 4, 1, 1])
#         # Spatial 정보를 압축하기 위해 global average pooling
#         x = torch.mean(x, dim=(3, 4))
#         #print(x.shape) #torch.Size([16, 5, 4])
#         #print(self.Softmax(x.mean(1)).shape) #torch.Size([16, 4])
#         x = x.mean(1)
#         #print(x.shape)
#         return self.Softmax(x)  # (B, num_experts)

class SEW_ResNet_MoE(nn.Module):
    def __init__(self, block, layers, num_classes=11, time_step=5, num_experts=4, DTA_ON=True, dvs=None):
        super(SEW_ResNet_MoE, self).__init__()
        self.dvs = dvs     
        self.T = time_step  
        self.num_experts = num_experts
        norm_layer = tdBatchNorm
        self._norm_layer = norm_layer
        self.inplanes = 32

        self.dilation = 1
        replace_stride_with_dilation = [False, False, False, False, False, False, False]
        self.groups = 1
        self.base_width = 32

        if self.dvs:
            self.input_conv = tdLayer(nn.Conv2d(2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(self.inplanes))
        else:
            self.input_conv = tdLayer(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(self.inplanes))

        if DTA_ON==True:
            self.attention = DTA(T=self.T, out_channels = 32)
        else: 
            self.attention = None
        self.MP = tdLayer(nn.MaxPool2d(2, 2))

        # 공유된 네트워크 (layer1 ~ layer6)
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer3 = self._make_layer(block, 32, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[2])
        self.layer4 = self._make_layer(block, 32, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[3])
        self.layer5 = self._make_layer(block, 32, layers[4], stride=1,
                                       dilate=replace_stride_with_dilation[4])
        self.layer6 = self._make_layer(block, 32, layers[5], stride=1,
                                       dilate=replace_stride_with_dilation[5])
        
        # Experts: 마지막 block과 Linear classifier를 포함
        # 각 expert는 Conv-BN-LIF block을 포함할 수 있도록 구성
        self.expert_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_layer(block, 32, layers[6], stride=1,
                                       dilate=replace_stride_with_dilation[6]),  # Expert별 마지막 block
                #tdLayer(nn.AdaptiveMaxPool2d((1, 1))),      # Adaptive Pooling을 expert 내부에만 적용
                nn.Flatten(start_dim=2),
                tdLayer(nn.Linear(32, num_classes))         # Expert별 Linear classifier
            ) for _ in range(num_experts)
        ])

        # Gate Network (Router) : shared feature의 정보를 사용하여 각 expert 가중치 결정
        # 여기서 shared feature는 shared_pool 이후 feature를 사용
        self.gate_network = MoEGateNetwork(in_channels=self.inplanes, num_experts=num_experts)

        self.LIF = LIFSpike()
        if DTA_ON:
            self.encoding = DTA(T=self.T, out_channels=32)
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
        # x: (B, T, C, H, W) 형태
        x = self.input_conv(x)          # (B, T, C, H, W)
        out_s = self.LIF(x)
        x = self.attention(x, out_s)           # Attention 적용
        x = self.MP(x)
        # Shared layers (B, T, C, H, W)
        x = self.layer1(x)
        x = self.MP(x)
        x = self.layer2(x)
        x = self.MP(x)
        x = self.layer3(x)
        x = self.MP(x)
        x = self.layer4(x)
        x = self.MP(x)
        x = self.layer5(x)
        x = self.MP(x)
        x = self.layer6(x)
        x = self.MP(x)
        # Gate Network 입력: 여기서 spatial 정보 그대로 유지
        gate_input = x  # (B, T, C, H', W')
        # 평균 over time (T) 차원을 적용하거나, 필요시 다른 방법 사용
        gate_input = gate_input  # (B, T, C, H', W')
        expert_weights = self.gate_network(gate_input)  # (B, num_experts)
        #print(expert_weights.shape) # (B, num_experts)
        # 각 Expert에 동일한 shared feature x를 넣고, Expert 별로 결과 도출

        #print("Expert 선택 확률 평균:", expert_weights.mean(dim=0))

        expert_outputs = []
        #print(x.shape) #torch.Size([16, 5, 32, 1, 1])
        for expert in self.expert_blocks:
            out = expert(x)  # (B, T, num_classes)
            expert_outputs.append(out.mean(1))
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, num_classes
        # Gate Network의 가중치를 이용하여 최종 출력 계산
        final_output = torch.sum(expert_weights.unsqueeze(-1) * expert_outputs, dim=1)  # (B, num_classes)
        return final_output

    def forward(self, x):
        return self._forward_impl(x)
    
class SEW_ResNet_CIFAR(nn.Module): #baseline dvs_cifar10
    def __init__(self, block, layers, num_classes=10, time_step=5, num_experts=4, DTA_ON=True, dvs=None):
        super(SEW_ResNet_CIFAR, self).__init__()
        self.dvs = dvs     
        self.T = time_step  
        self.num_experts = num_experts
        norm_layer = tdBatchNorm
        self._norm_layer = norm_layer
        self.inplanes = 64

        self.dilation = 1
        replace_stride_with_dilation = [False, False, False, False, False, False, False, False]
        self.groups = 1
        self.base_width = 64

        if self.dvs:
            self.input_conv = tdLayer(nn.Conv2d(2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(self.inplanes))
        else:
            self.input_conv = tdLayer(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(self.inplanes))

        if DTA_ON==True:
            self.attention = DTA(T=self.T, out_channels = 64)
            self.attention2 = DTA(T=self.T, out_channels = 128)
        else: 
            self.attention = None
        self.MP = tdLayer(nn.MaxPool2d(2, 2))

        # 공유된 네트워크
        
        #c64k3s1-BN-PLIF-{SEW Block (c64)-MPk2s2}*4-c128k3s1-
        #BN-PLIF-{SEW Block (c128)-MPk2s2}*3-FC10 (Wide-7B-Net)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[2])
        self.layer4 = self._make_layer(block, 64, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[3])
        self.layer5 = self._make_layer(block, 128, layers[4], stride=1,
                                       dilate=replace_stride_with_dilation[4])
        self.layer6 = self._make_layer(block, 128, layers[5], stride=1,
                                       dilate=replace_stride_with_dilation[5])
        self.layer7 = self._make_layer(block, 128, layers[6], stride=1,
                                       dilate=replace_stride_with_dilation[6])
        self.layer8 = self._make_layer(block, 128, layers[7], stride=1,
                                       dilate=replace_stride_with_dilation[7])

        self.LIF = LIFSpike()
        
        self.fc = tdLayer(nn.Linear(128, num_classes, bias=True))

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
        # x: (B, T, C, H, W) 형태
        x = self.input_conv(x)          # (B, T, C, H, W)
        x = self.LIF(x)
        x = self.layer1(x) #C64
        x = self.MP(x)      
        x = self.layer2(x) #C64
        #x = self.MP(x)  #resize 48 할때 주석 #64할때 주석
        x = self.layer3(x) #C64
        x = self.MP(x)      
        x = self.layer4(x) #C64
        x = self.MP(x)  #resize 48 할때 주석
        x = self.layer5(x) #C128
        x = self.layer6(x) #C128
        x = self.MP(x) 
        x = self.layer7(x) #C128
        x = self.MP(x) 
        x = self.layer8(x) #C128
        x = self.MP(x) 
        x = x.flatten(2)
        #x = self.fc(x.mean(1))
        x = self.fc(x)
        return x.mean(1)    

    def forward(self, x):
        return self._forward_impl(x)
    

class SEW_ResNet_GA_CIFAR(nn.Module): #GAGAGAAAGAAGAGAGA
    def __init__(self, block, layers, num_classes=10, time_step=5, num_experts=4, DTA_ON=True, dvs=None):
        super(SEW_ResNet_GA_CIFAR, self).__init__()
        self.dvs = dvs     
        self.T = time_step  
        self.num_experts = num_experts
        norm_layer = tdBatchNorm
        self._norm_layer = norm_layer
        self.inplanes = 64

        self.dilation = 1
        replace_stride_with_dilation = [False, False, False, False, False, False, False, False]
        self.groups = 1
        self.base_width = 64

        if self.dvs:
            self.input_conv = tdLayer(nn.Conv2d(2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(self.inplanes))
        else:
            self.input_conv = tdLayer(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(self.inplanes))

        if DTA_ON==True:
            self.attention = DTA(T=self.T, out_channels = 64)
            self.attention2 = DTA(T=self.T, out_channels = 128)
        else: 
            self.attention = None
        self.MP = tdLayer(nn.MaxPool2d(2, 2))

        # 공유된 네트워크
        
        #c64k3s1-BN-PLIF-{SEW Block (c64)-MPk2s2}*4-c128k3s1-
        #BN-PLIF-{SEW Block (c128)-MPk2s2}*3-FC10 (Wide-7B-Net)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[2])
        self.layer4 = self._make_layer(block, 64, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[3])
        self.layer5 = self._make_layer(block, 128, layers[4], stride=1,
                                       dilate=replace_stride_with_dilation[4])
        #self.layer5 =  tdLayer(nn.Conv2d(128, 128, kernel_size=3, stride=1, 
        #                                 padding=1, bias=False), norm_layer(128))
        self.layer6 = self._make_layer(block, 128, layers[5], stride=1,
                                       dilate=replace_stride_with_dilation[5])
        self.layer7 = self._make_layer(block, 128, layers[6], stride=1,
                                       dilate=replace_stride_with_dilation[6])
        self.layer8 = self._make_layer(block, 128, layers[7], stride=1,
                                       dilate=replace_stride_with_dilation[7])

        self.LIF = LIFSpike()
        
        self.fc = tdLayer(nn.Linear(128, num_classes, bias=True))

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
        # x: (B, T, C, H, W) 형태
        x = self.input_conv(x)          # (B, T, C, H, W)
        out_s = self.LIF(x) #for GA
        x = self.attention(x, out_s) #for GA
        x = self.MP(x) #for GA #nofirstmp면 주석
        x = self.layer1(x) #C64
        x = self.MP(x)      #0218 GAtest주석 nofirstmp면 주석 해제
        x = self.layer2(x) #C64
        #x = self.MP(x)  #resize 48 할때 주석 #64할때 주석
        x = self.layer3(x) #C64
        x = self.MP(x)      #0218 GAtest주석
        x = self.layer4(x) #C64
        x = self.MP(x)  #resize 48 할때 주석
        x = self.layer5(x) #C128
        x = self.layer6(x) #C128
        x = self.MP(x) 
        x = self.layer7(x) #C128
        x = self.MP(x) 
        x = self.layer8(x) #C128
        #x = self.MP(x)
        x = x.flatten(2)
        x = self.fc(x)
        return x.mean(1)    


    def forward(self, x):
        return self._forward_impl(x)


class SEW_ResNet_GA_MOE_CIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=10, time_step=5, num_experts=4, DTA_ON=True, dvs=None):
        super(SEW_ResNet_MoE_CIFAR, self).__init__()
        self.dvs = dvs     
        self.T = time_step  
        self.num_experts = num_experts
        norm_layer = tdBatchNorm
        self._norm_layer = norm_layer
        self.inplanes = 64

        self.dilation = 1
        replace_stride_with_dilation = [False, False, False, False, False, False, False, False]
        self.groups = 1
        self.base_width = 64

        if self.dvs:
            self.input_conv = tdLayer(nn.Conv2d(2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(self.inplanes))
        else:
            self.input_conv = tdLayer(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(self.inplanes))

        if DTA_ON==True:
            self.attention = DTA(T=self.T, out_channels = 64)
        else: 
            self.attention = None
        self.MP = tdLayer(nn.MaxPool2d(2, 2))

        # 공유된 네트워크
        
        #c64k3s1-BN-PLIF-{SEW Block (c64)-MPk2s2}*4-c128k3s1-
        #BN-PLIF-{SEW Block (c128)-MPk2s2}*3-FC10 (Wide-7B-Net)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[2])
        self.layer4 = self._make_layer(block, 64, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[3])
        self.layer5 = self._make_layer(block, 128, layers[4], stride=1,
                                       dilate=replace_stride_with_dilation[4])
        self.layer6 = self._make_layer(block, 128, layers[5], stride=1,
                                       dilate=replace_stride_with_dilation[5])
        self.layer7 = self._make_layer(block, 128, layers[6], stride=1,
                                       dilate=replace_stride_with_dilation[6])

        # 여기에서는 중간 Feature를 downsampling 하는 방식으로 MaxPool 사용
        self.shared_pool = tdLayer(nn.MaxPool2d(2, 2))
        
        # Experts: 마지막 block과 Linear classifier를 포함
        # 각 expert는 Conv-BN-LIF block을 포함할 수 있도록 구성
        self.expert_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_layer(block, 128, layers[7], stride=1,
                                       dilate=replace_stride_with_dilation[7]),  # Expert별 마지막 block
                #tdLayer(nn.AdaptiveMaxPool2d((1, 1))),      # Adaptive Pooling을 expert 내부에만 적용
                nn.Flatten(start_dim=2),
                tdLayer(nn.Linear(128, num_classes))         # Expert별 Linear classifier
            ) for _ in range(num_experts)
        ])

        # Gate Network (Router) : shared feature의 정보를 사용하여 각 expert 가중치 결정
        # 여기서 shared feature는 shared_pool 이후 feature를 사용
        self.gate_network = MoEGateNetwork(in_channels=self.inplanes, num_experts=num_experts)

        self.LIF = LIFSpike()
        
        
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
        # x: (B, T, C, H, W) 형태
        x = self.input_conv(x) #C64       # (B, T, C, H, W)
        out_s = self.LIF(x)
        x = self.attention(x, out_s)           # Attention 적용
        x = self.MP(x) #64
        # Shared layers (B, T, C, H, W)
        x = self.layer1(x) #C64
        x = self.MP(x) #32
        x = self.layer2(x) #C64
        #x = self.MP(x) #16
        x = self.layer3(x) #C64
        x = self.MP(x) #8
        x = self.layer4(x) #C64
        x = self.MP(x) #4
        x = self.layer5(x) #C128
        #x = self.MP(x) #2
        x = self.layer6(x) #C128
        x = self.MP(x) #2
        x = self.layer7(x) #C128
        x = self.MP(x) #1
        # Gate Network 입력: 여기서 spatial 정보 그대로 유지
        gate_input = x  # (B, T, C, H', W')
        # 평균 over time (T) 차원을 적용하거나, 필요시 다른 방법 사용
        gate_input = gate_input  # (B, T, C, H', W')
        expert_weights = self.gate_network(gate_input)  # (B, num_experts)
        #print(expert_weights.shape) # (B, num_experts)
        # 각 Expert에 동일한 shared feature x를 넣고, Expert 별로 결과 도출

        #print("Expert 선택 확률 평균:", expert_weights.mean(dim=0))

        expert_outputs = []
        for expert in self.expert_blocks:
            out = expert(x)  # (B, T, num_classes)
            expert_outputs.append(out.mean(1))
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, num_classes
        # Gate Network의 가중치를 이용하여 최종 출력 계산
        final_output = torch.sum(expert_weights.unsqueeze(-1) * expert_outputs, dim=1)  # (B, num_classes)
        return final_output

    def forward(self, x):
        return self._forward_impl(x)

def GA_sewresnet_moe(**kwargs):
    return SEW_ResNet_MoE(BasicBlock_SEW, [1, 1, 1, 1, 1, 1, 1], **kwargs)

def sewresnet_ga_moe_cifar(**kwargs):
    return SEW_ResNet_GA_MoE_CIFAR(BasicBlock_SEW, [1, 1, 1, 1, 1, 1, 1, 1], **kwargs)

def sewresnet_ga_cifar(**kwargs):
    return SEW_ResNet_GA_CIFAR(BasicBlock_SEW, [1, 1, 1, 1, 1, 1, 1, 1], **kwargs)

def sewresnet_cifar(**kwargs):
    return SEW_ResNet_CIFAR(BasicBlock_SEW, [1, 1, 1, 1, 1, 1, 1, 1], **kwargs)




