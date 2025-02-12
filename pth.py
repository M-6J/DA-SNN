import torch
import torch.nn as nn
from models.DTA_SNN import *
#from models.SEW_ResNet import *
from models.gesture import *


# 모델 객체 생성
model = MSResNet()

# 저장된 가중치 로드
model.load_state_dict(torch.load("D:\DA-SNN\wan_TCJA_C3x3_avgpool_dvs_gesture-MSResNet-S42-B16-T5-E150-LR0.001.pth.tar", map_location=torch.device("cpu")))

# 전체 네트워크 출력
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params / 1e6:.2f} M")  # 백만 개 단위로 변환

