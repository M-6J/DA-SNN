import torch
import torch.nn as nn
from models.DTA_SNN import *
#from models.SEW_ResNet import *
#from models.gesture import *
#from models.MS_ResNet import *
from models.SEW_ResNet_moe import *
from models.layers import *
from spikingjelly.activation_based.neuron import ParametricLIFNode

# 모델 객체 생성
model = sewresnet_ga_moe_cifar_v2(num_classes=10, time_step=10, DTA_ON=True, dvs=True)
#model = VGGNet()
# 저장된 가중치 로드
#model.load_state_dict(torch.load("/home/lee/DA-SNN/DVSCIFAR10-SEWResNet-resize64x64_GA_MOE_fix1_fc_layer678_aug-S42-B16-T10-E200-LR0.001.pth.tar", map_location=torch.device("cpu")))

# 전체 네트워크 출력
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params / 1e6:.2f} M")  # 백만 개 단위로 변환


# from calflops import calculate_flops

# model = dta_sewresnet_moe(num_classes=11, time_step=20, DTA_ON=True, dvs=True)
# input_shape = (16, 20, 2, 128, 128)

# flops, macs, params = calculate_flops(model=model, 
#                                       input_shape=input_shape,
#                                       output_as_string=True,
#                                       output_precision=4)
# print("ResNet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))