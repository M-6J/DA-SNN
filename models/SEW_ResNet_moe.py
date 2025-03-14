from models.layers import *
from models.DTA import DTA
from models.TXA import T_XA_128
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based.neuron import ParametricLIFNode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #for conv3d 채널기반
# class MoEGateNetwork(nn.Module):
#     def __init__(self, in_channels, num_experts):
#         super().__init__()
#         # self.conv1 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         # self.bn1 = nn.BatchNorm3d(128)
#         # self.conv2 = nn.Conv3d(128, num_experts, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         # self.bn2 = nn.BatchNorm3d(num_experts)  
#         # self.ReLU = nn.ReLU()
#         # self.Softmax = nn.Softmax(dim=1)
#         # self.mxpool = nn.AdaptiveMaxPool3d((None, 1, 1))  
#         # self.avpool = nn.AdaptiveAvgPool3d((None, 1, 1))  
        
#         # for add last linear
#         self.conv1 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.bn1 = nn.BatchNorm3d(256)
#         self.conv2 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.bn2 = nn.BatchNorm3d(256)  
#         self.ReLU = nn.ReLU()
#         self.Softmax = nn.Softmax(dim=1)
#         self.mxpool = nn.AdaptiveMaxPool3d((None, 1, 1))  
#         self.avpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))  
#         self.fc = tdLayer(nn.Linear(256, num_experts))

#     def forward(self, x):
#         # 기존 x.shape: (B, T, C, H, W)
#         x = x.permute(0, 2, 1, 3, 4)  #Conv3D는 (B, C, T, H, W) 입력 형식이므로 변환
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.ReLU(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
        
#         # for add last linear
#         x = x.permute(0,2,1,3,4)
#         x = self.avpool(x)
#         x = torch.flatten(x, 2)
#         logits = self.fc(x)
#         logits = logits.mean(dim=1)
#         return self.Softmax(logits)
        
        
#         # Global Average Pooling (시간 + 공간 차원 압축)
#         #x = torch.mean(x, dim=(3, 4))  # (B, num_experts, T) → 공간 차원(H, W) 평균
#         #x = self.avpool(x)
#         #x = torch.flatten(x, 2)
#         #x = x.mean(dim=2)  # (B, num_experts) → 시간 차원(T) 평균

#         #return self.Softmax(x)  # Expert 선택 확률 출력

# class TemporalMean(nn.Module):
#     def forward(self, x):
#         # x: (B, T, C)
#         return x.mean(dim=1)  # (B, C)

# class MoEGateNetwork(nn.Module): #conv2D
#     def __init__(self, in_channels, num_experts):
#         super().__init__()
#         # 여기서는 Conv 기반으로 설계해서, shared feature의 spatial 정보를 유지
#         norm_layer = tdBatchNorm
#         self.conv1 = tdLayer(nn.Conv2d(128, 128, kernel_size=3, padding=1), norm_layer(128))
#         self.conv2 = tdLayer(nn.Conv2d(128, num_experts, kernel_size=3, padding=1), norm_layer(num_experts))
#         self.ReLU = tdLayer(nn.ReLU())
#         self.Softmax = nn.Softmax(dim=1)
#         self.avpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))
#         self.mxpool = tdLayer(nn.AdaptiveMaxPool2d((1, 1)))
        
#     def forward(self, x):
#        #print(x.shape) torch.Size([16, 5, 32, 1, 1])
#         x = self.conv1(x)
#         x = self.ReLU(x)
#         x = self.conv2(x)
        
#         # Spatial 정보를 압축하기 위해 global average pooling
#         x = torch.mean(x, dim=(3, 4))
        
#         #x = self.mxpool(x)
#         #x = torch.flatten(x, 2)
#         x = x.mean(1)
#         return self.Softmax(x)  # (B, num_experts)

# class MoEGateNetwork(nn.Module): #Linear
#     def __init__(self, in_channels, num_experts):
#         super().__init__()
#         # self.fc1 = tdLayer(nn.Linear(in_channels, 128))  # 중간 차원 축소
#         # self.relu = nn.ReLU()
#         # self.fc2 = tdLayer(nn.Linear(128, 128))
#         # self.fc3 = tdLayer(nn.Linear(128, num_experts))  # Expert 선택을 위한 출력
#         # self.avpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1))) 
#         # self.softmax = nn.Softmax(dim=1)
        
#         self.fc1 = nn.Linear(in_channels, 128)  # 중간 차원 축소
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, num_experts)  # Expert 선택을 위한 출력
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         # 기존 x.shape: (B, T, C, H, W)
#         x = x.mean(dim=(3, 4))  # 공간 차원(H, W) 평균 풀링 → (B, T, C)
        
#         #x = self.avpool(x)
#         #x = torch.flatten(x, 2)
#         #x = x.mean(dim=1)  # 시간 차원(T) 평균 풀링 → (B, C)

#         x = self.fc1(x)  # Fully Connected Layer 1
#         x = self.relu(x)
#         x = self.fc2(x)  # Expert 선택 확률 계산
#         x = self.relu(x)
#         x = self.fc3(x)
        
#         x = x.mean(dim=1)

#         return self.softmax(x)  # (B, num_experts) 형태의 확률 출력



class MoEGateNetwork(nn.Module):  
    def __init__(self, in_channels, num_experts=4, top_k=1, noise_std=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std  # Gaussian Noise Standard Deviation

        # 🔹 Fully Connected Router
        self.fc1 = nn.Linear(128, 128)  #<--a100에서 마지막 돌린건 in_channels로 받았는데 그 값이 64임
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_experts)  # Expert 선택을 위한 출력

        # 🔹 Xavier Initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier Initialization 적용"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 기존 x.shape: (B, T, C, H, W)
        x = x.mean(dim=(3, 4))  # 공간 차원(H, W) 평균 풀링 → (B, T, C)       
        x = x.mean(dim=1)  # 시간 차원(T) 평균 풀링 → (B, C)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        gate_logits = self.fc3(x)  # (B, num_experts)

        # 🔹 Gaussian Noise 추가 (Noisy Top-K)
        if self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise

        # 🔹 Gate 확률 계산 (Softmax)
        gate_probs = F.softmax(gate_logits, dim=1)  # (B, num_experts)

        # 🔹 Load Balancing Loss 계산
        gate_probs_mean = gate_probs.mean(dim=0)  # (num_experts,)
        expected_prob = 1.0 / self.num_experts
        load_balance_loss = ((gate_probs_mean - expected_prob) ** 2).mean() * self.num_experts

        # 🔹 Top-K Expert 선택
        topk_probs, topk_indices = torch.topk(gate_probs, self.top_k, dim=1)  # (B, top_k)

        return gate_probs, topk_probs, topk_indices, load_balance_loss  # Expert 선택 확률 + Loss 반환



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
        expert_weights = self.gate_network(gate_input, )  # (B, num_experts)
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
        
        self.fc = nn.Linear(128, num_classes, bias=True)

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
        x = self.fc(x.mean(1))
        #x = self.fc(x)
        return x    

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
        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))
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
        #self.layer8 = tdLayer(nn.Conv2d(128, 128, 1), norm_layer(128))
        #self.layer8 = tdLayer(nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=1, bias=False), norm_layer(128))
        self.LIF = LIFSpike()
        
        self.fc = tdLayer(nn.Linear(128, num_classes, bias=True))
        #self.fc = nn.Linear(128, num_classes, bias=True)
        # self.MLPfc = nn.Sequential(
        #     nn.Linear(128, 256, bias=True),  
        #     LIFSpike(),  # 스파이킹 뉴런 추가
        #     nn.Linear(256, num_classes, bias=True)
        # )


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
        #x = self.LIF(x)
        out_s = self.LIF(x) #for GA
        x = self.attention(x, out_s) #for GA
        x = self.MP(x) #for GA #nofirstmp면 주석
        x = self.layer1(x) #C64
        #x = self.MP(x)      #0218 GAtest주석 nofirstmp면 주석 해제
        x = self.layer2(x) #C64
        #x = self.MP(x)  #resize 48 할때 주석 #64할때 주석
        x = self.layer3(x) #C64
        #x = self.MP(x)      #0218 GAtest주석
        x = self.layer4(x) #C64
        x = self.MP(x)  #resize 48 할때 주석
        x = self.layer5(x) #C128
        x = self.MP(x) 
        x = self.layer6(x) #C128
        x = self.MP(x) 
        x = self.layer7(x) #C128
        x = self.MP(x) 
        x = self.layer8(x) #C128
        x = self.MP(x)
        #x = self.LIF(x)
        #x = self.avgpool(x)
        #print(x.shape)
        x = x.flatten(2)
        x = self.fc(x)
        #x = self.MLPfc(x.mean(1))
        return x.mean(1)


    def forward(self, x):
        return self._forward_impl(x)

# SNN MoE Forward with Top-2 Gating
def forward_moe(x, gate_network, expert_blocks, num_experts):
    """
    x: shared feature input to experts, shape (B, T, C, H, W)
    gate_network: instance of MoEGateNetwork, returns expert_weights (B, num_experts)
    expert_blocks: list (ModuleList) of expert modules, each outputs (B, T, num_classes)
    num_experts: int, number of experts (here 4)
    """
    B = x.size(0)
    
    # 1. Compute expert selection probabilities from the router
    expert_weights = gate_network(x)  # (B, num_experts)
    # For debugging:
    # print("Expert 선택 확률 평균:", expert_weights.mean(dim=0))
    
    # 2. Top-1 selection: deterministically select the expert with highest probability
    top1_indices = torch.argmax(expert_weights, dim=1)  # (B,)
    
    # 3. For each sample, remove Top-1 from probability vector and sample Top-2 from the remaining experts
    mask = torch.ones_like(expert_weights)
    mask.scatter_(1, top1_indices.unsqueeze(1), 0)  # Set Top-1 positions to 0
    remaining_weights = expert_weights * mask  # (B, num_experts)
    # Normalize remaining weights per sample
    remaining_weights_sum = remaining_weights.sum(dim=1, keepdim=True) + 1e-8
    normalized_remaining = remaining_weights / remaining_weights_sum  # (B, num_experts)
    
    # Top-2 selection: sample one expert from remaining experts for each sample
    top2_indices = []
    for b in range(B):
        # Multinomial sampling: sample one index from remaining experts based on normalized weights
        idx = torch.multinomial(normalized_remaining[b], num_samples=1)
        top2_indices.append(idx)
    top2_indices = torch.cat(top2_indices, dim=0)  # (B,)
    
    # 4. Create new expert weight vector: only Top-1 and Top-2 retain their original weights, then renormalize
    selected_weights = []
    for b in range(B):
        w_top1 = expert_weights[b, top1_indices[b]]
        w_top2 = expert_weights[b, top2_indices[b]]
        total = w_top1 + w_top2 + 1e-8
        # Create a zero vector and assign normalized weights at Top-1 and Top-2 positions
        new_weight = torch.zeros_like(expert_weights[b])
        new_weight[top1_indices[b]] = w_top1 / total
        new_weight[top2_indices[b]] = w_top2 / total
        selected_weights.append(new_weight.unsqueeze(0))
    selected_weights = torch.cat(selected_weights, dim=0)  # (B, num_experts)
    
    # 5. Compute each Expert's output
    expert_outputs = []
    for expert in expert_blocks:
        # Each expert: output shape (B, T, num_classes)
        out = expert(x)
        # Average over time dimension to get (B, num_classes)
        expert_outputs.append(out.mean(dim=1))
    expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, num_classes)
    
    # 6. Combine the outputs of the selected Experts using the selected weights
    final_output = torch.sum(selected_weights.unsqueeze(-1) * expert_outputs, dim=1)  # (B, num_classes)
    
    return final_output

class SEW_ResNet_GA_MOE_CIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=10, time_step=10, num_experts=4, DTA_ON=True, dvs=None):
        super(SEW_ResNet_GA_MOE_CIFAR, self).__init__()
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
        
        dropout_probs = [0.1, 0.15, 0.2, 0.25]

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
        # self.layer6 = self._make_layer(block, 128, layers[5], stride=1,
        #                                dilate=replace_stride_with_dilation[5])
        # self.layer7 = self._make_layer(block, 128, layers[6], stride=1,
        #                                dilate=replace_stride_with_dilation[6])

        
        # Experts: 마지막 block과 Linear classifier를 포함
        # 각 expert는 Conv-BN-LIF block을 포함할 수 있도록 구성
        self.expert_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_layer(block, 128, layers[5], stride=1,
                                        dilate=replace_stride_with_dilation[5]),
                #T_XA_128(self.T) if idx == 1 else nn.Identity(),
                #T_XA_128(self.T) if idx in [1, 2] else nn.Identity(),
                tdLayer(nn.MaxPool2d(2, 2)),
                self._make_layer(block, 128, layers[6], stride=1,
                                       dilate=replace_stride_with_dilation[6]),
                #T_XA_128(self.T) if idx == 2 else nn.Identity(),
                #T_XA_128(self.T) if idx in [2, 3] else nn.Identity(),
                tdLayer(nn.MaxPool2d(2, 2)),
                self._make_layer(block, 128, layers[7], stride=1,
                                       dilate=replace_stride_with_dilation[7]),  
                #T_XA_128(self.T) if idx == 3 else nn.Identity(),
                tdLayer(nn.MaxPool2d(2, 2)),  
                #nn.Flatten(start_dim=2),           #share fc 에서 주석
                nn.Dropout(p=dropout_probs[idx]),
                #tdLayer(nn.Linear(128, num_classes)) #share fc 에서 주석
                #TemporalMean(),
                #nn.Linear(128, num_classes)
            ) for idx, _ in enumerate(range(num_experts))
        ])
        self.shared_classifier = tdLayer(nn.Linear(128, num_classes))

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
        #x = self.MP(x) #32
        x = self.layer2(x) #C64
        #x = self.MP(x) #16
        x = self.layer3(x) #C64
        #x = self.MP(x) #8
        x = self.layer4(x) #C64
        x = self.MP(x) #4
        x = self.layer5(x) #C128
        x = self.MP(x) #2
        #x = self.layer6(x) #C128
        #x = self.MP(x) #2
        #x = self.layer7(x) #C128
        #x = self.MP(x) #1
        # Gate Network 입력: 여기서 spatial 정보 그대로 유지
        gate_input = x  # (B, T, C, H', W')
        # 평균 over time (T) 차원을 적용하거나, 필요시 다른 방법 사용
        gate_input = gate_input  # (B, T, C, H', W')
        expert_weights = self.gate_network(gate_input)  # (B, num_experts)
        #print(expert_weights.shape) # (B, num_experts)
        # 각 Expert에 동일한 shared feature x를 넣고, Expert 별로 결과 도출

        #print("Expert 선택 확률 평균:", expert_weights.mean(dim=0))
     
        # expert_outputs = []
        # for expert in self.expert_blocks:
        #     out = expert(x)  # (B, T, num_classes)
        #     #out = F.softmax(out, dim=-1)  # 🚨 각 Expert의 Logits을 정규화
        #     #out = F.normalize(out, p=2, dim=-1) #L2 normalization
        #     #expert_outputs.append(out) #use fc
        #     expert_outputs.append(out.mean(1)) #use tdfc
        
        # expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, num_classes
        # # Gate Network의 가중치를 이용하여 최종 출력 계산
        # final_output = torch.sum(expert_weights.unsqueeze(-1) * expert_outputs, dim=1)  # (B, num_classes)
        # return final_output
        
        ### for shared fc ###
        expert_features = []
        for expert in self.expert_blocks:
            out = expert(x)  # (B, T, 128, H, W)
            out = out.mean(dim=[3, 4])  # (B, T, 128) 🔥 공간 차원(H, W) 평균
            expert_features.append(out)
        expert_features = torch.stack(expert_features, dim=1)  # (B, num_experts, T, 128)
        # 🔥 Shared Classifier 적용
        expert_logits = self.shared_classifier(expert_features)  # (B, num_experts, T, num_classes)
        # 🔥 Time Dimension(T) 평균 (선택적)
        expert_logits = expert_logits.mean(2)  # (B, num_experts, num_classes)
        # 🔥 MoE 가중합 적용
        final_output = torch.sum(expert_weights.unsqueeze(-1) * expert_logits, dim=1)  # (B, num_classes)
        return final_output
        
        # # ### for top k ###
        # # Apply top-2 gating on expert_weights and combine expert outputs:
        # final_output = forward_moe(gate_input, self.gate_network, self.expert_blocks, num_experts=self.num_experts)
        
        # return final_output
        
    def forward(self, x):
        return self._forward_impl(x)


class SEW_ResNet_GA_MOE_CIFAR_V2(nn.Module):
    def __init__(self, block, layers, num_classes=10, time_step=10, num_experts=4, DTA_ON=True, dvs=None):
        super(SEW_ResNet_GA_MOE_CIFAR_V2, self).__init__()
        self.dvs = dvs     
        self.T = time_step  
        self.num_experts = num_experts
        self.num_classes = num_classes
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
        
        dropout_probs = [0.1, 0.15, 0.2, 0.25]
        
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

        self.expert_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_layer(block, 128, layers[5], stride=1,
                                        dilate=replace_stride_with_dilation[5]),
                tdLayer(nn.MaxPool2d(2, 2)),
                self._make_layer(block, 128, layers[6], stride=1,
                                       dilate=replace_stride_with_dilation[6]),
                tdLayer(nn.MaxPool2d(2, 2)),
                self._make_layer(block, 128, layers[7], stride=1,
                                       dilate=replace_stride_with_dilation[7]),  
                tdLayer(nn.MaxPool2d(2, 2)),  
                nn.Dropout(p=dropout_probs[idx]),
            ) for idx, _ in enumerate(range(num_experts))
        ])
        self.shared_classifier = tdLayer(nn.Linear(128, num_classes))

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
        B, T, C, H, W = x.shape

        # 🔹 Backbone 처리
        x = self.input_conv(x)  
        out_s = self.LIF(x)
        x = self.attention(x, out_s)           
        x = self.MP(x) 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.MP(x) 
        x = self.layer5(x)
        x = self.MP(x) 

        # 🔹 Router (Gate Network) 실행
        gate_input = x  # (B, T, C, H', W')
        expert_weights, topk_probs, topk_indices, load_balance_loss = self.gate_network(gate_input)

        # 🔹 Experts 실행
        expert_features = []
        for expert in self.expert_blocks:
            out = expert(x)  # (B, T, 128, H, W)
            out = out.mean(dim=[3, 4])
            expert_features.append(out)

        expert_features = torch.stack(expert_features, dim=1)  # (B, num_experts, T, 128, H, W)
        expert_logits = self.shared_classifier(expert_features)  # (B, num_experts, T, num_classes)

        # 🔹 선택된 Experts의 Logits만 활용
        selected_expert_logits = torch.gather(
            expert_logits,
            1,
            topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, expert_logits.shape[2], self.num_classes)
        )

        # 🔹 최종 출력 계산 (시간 차원 유지)
        final_output = torch.einsum('bk,bktd->btd', topk_probs, selected_expert_logits).mean(dim=1)  # (B, num_classes)

        return final_output, load_balance_loss

        
    def forward(self, x):
        return self._forward_impl(x)


def GA_sewresnet_moe(**kwargs):
    return SEW_ResNet_MoE(BasicBlock_SEW, [1, 1, 1, 1, 1, 1, 1], **kwargs)

def sewresnet_ga_moe_cifar(**kwargs):
    return SEW_ResNet_GA_MOE_CIFAR(BasicBlock_SEW, [1, 1, 1, 1, 1, 1, 1, 1], **kwargs)

def sewresnet_ga_moe_cifar_v2(**kwargs):
    return SEW_ResNet_GA_MOE_CIFAR_V2(BasicBlock_SEW, [1, 1, 1, 1, 1, 1, 1, 1], **kwargs)
        

def sewresnet_ga_cifar(**kwargs):
    return SEW_ResNet_GA_CIFAR(BasicBlock_SEW, [1, 1, 1, 1, 1, 1, 1, 1], **kwargs)

def sewresnet_cifar(**kwargs):
    return SEW_ResNet_CIFAR(BasicBlock_SEW, [1, 1, 1, 1, 1, 1, 1, 1], **kwargs)




