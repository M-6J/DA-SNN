from models.TXA import T_XA
from models.TNA import T_NA

import torch.nn as nn


# class DTA(nn.Module):  # with MoE @@@수정중@@@
#     def __init__(self, T, out_channels):
#         super().__init__()

#         self.T_NA = T_NA(in_planes=out_channels * T, kernel_size=7)
#         self.T_XA = T_XA(time_step=T)

#         # 게이트 메커니즘을 위한 FC 레이어
#         self.gate_fc = nn.Sequential(
#             nn.Linear(T * out_channels, 256),  # H, W 차원 제거 후 크기를 줄임
#             nn.ReLU(),
#             nn.Linear(256, 2),  # 두 전문가에 대한 가중치 (T-XA와 T-NA)
#             nn.Softmax(dim=1)  # 확률 분포로 정규화
#         )
        
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x_seq, spikes):
#         B, T, C, H, W = x_seq.shape

#         # T-NA 계산
#         x_seq_2 = x_seq.reshape(B, T * C, H, W)
#         T_NA = self.T_NA(x_seq_2)
#         T_NA = T_NA.reshape(B, T, C, H, W)

#         # T-XA 계산
#         T_XA = self.T_XA(x_seq)

#         # 게이트 입력 준비 (H, W를 평균으로 압축)
#         gate_input = x_seq.mean(dim=[3, 4]).reshape(B, -1)
#         gate_weights = self.gate_fc(gate_input)
#         gate_weights = gate_weights.view(B, 2, 1, 1, 1)

#         # 전문가 조합
#         out = gate_weights[:, 0:1] * T_XA + gate_weights[:, 1:2] * T_NA
#         out = self.sigmoid(out)

#         # 최종 출력
#         y_seq = out * spikes
#         return y_seq


# class DTA(nn.Module): #with MoE //Demo 버전
#     def __init__(self, T, out_channels):
#         super().__init__()

#         self.T_NA = T_NA(in_planes=out_channels*T, kernel_size=7)
#         self.T_XA = T_XA(time_step=T) 

#         # 게이트 메커니즘을 위한 레이어 추가
#         self.gate_fc = nn.Sequential(
#             nn.Linear(T * C * H * W, 256),  # 입력 데이터의 전체 크기를 고려하여 게이트 설계
#             nn.ReLU(),
#             nn.Linear(256, 2),  # T-XA와 T-NA에 대한 가중치를 출력 (2개)
#             nn.Softmax(dim=1)  # 두 전문가에 대한 가중치를 확률 분포로 출력
#         )

#         self.sigmoid = nn.Sigmoid()
   
#     def forward(self, x_seq, spikes):
        
#         B, T, C, H, W = x_seq.shape 
        
#         # T-NA 계산
#         x_seq_2 = x_seq.reshape(B, T*C, H, W)
#         T_NA = self.T_NA(x_seq_2) 
#         T_NA = T_NA.reshape(B, T, C, H, W)
        
#         # T-XA 계산
#         T_XA = self.T_XA(x_seq) 
        
#         # 게이트 메커니즘: 입력 데이터에 따라 T-XA와 T-NA의 가중치를 결정
#         gate_input = x_seq.reshape(B, -1)  # 입력 데이터를 평탄화
#         gate_weights = self.gate_fc(gate_input)  # 게이트가 T-XA와 T-NA에 대한 가중치를 출력
#         gate_weights = gate_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 차원 맞추기 (B, 2, 1, 1, 1)

#         # MoE 출력: 게이트 가중치에 따라 T-XA와 T-NA의 출력을 조합
#         out = gate_weights[:, 0:1] * T_XA + gate_weights[:, 1:2] * T_NA
#         out = self.sigmoid(out)  # 시그모이드 활성화 함수 적용

#         # 최종 출력
#         y_seq = out * spikes  

#         return y_seq


#Only T_XA
class DTA(nn.Module):
    def __init__(self, T, out_channels):
        super().__init__()
        self.T_XA = T_XA(time_step=T) 
        self.sigmoid = nn.Sigmoid()
   
    def forward(self, x_seq, spikes):
        B, T, C, H, W = x_seq.shape     
        
        T_XA = self.T_XA(x_seq) 
    
        out = T_XA
        y_seq = out * spikes  
        #print(y_seq.shape)

        return y_seq   


#  #Only T_NA
# class DTA(nn.Module):
#     def __init__(self, T, out_channels):
#         super().__init__()

#         self.T_NA = T_NA(in_planes=out_channels*T, kernel_size=7)

#         self.softmax = nn.Softmax(dim=1)
#         self.sigmoid = nn.Sigmoid()
   
#     def forward(self, x_seq, spikes):
        
#         T, B, C, H, W = x_seq.shape     
        
#         x_seq_2 = x_seq.reshape(B, T*C, H, W)
#         T_NA = self.T_NA(x_seq_2) 
#         T_NA = T_NA.reshape(B, T, C, H, W)
#         out = T_NA.reshape(T, B, C, H, W)
#         y_seq = out * spikes  

#         return y_seq  


#  #gesture
# class DTA(nn.Module):
#     def __init__(self, T, out_channels):
#         super().__init__()

#         self.T_NA = T_NA(in_planes=out_channels*T, kernel_size=7)
#         self.T_XA = T_XA(time_step=T) 

#         self.softmax = nn.Softmax(dim=1)
#         self.sigmoid = nn.Sigmoid()
   
#     def forward(self, x_seq, spikes):
        
#         T, B, C, H, W = x_seq.shape     
        
#         x_seq_2 = x_seq.reshape(B, T*C, H, W)
#         T_NA = self.T_NA(x_seq_2) 
#         T_NA = T_NA.reshape(T, B, C, H, W)
        
#         T_XA = self.T_XA(x_seq) 
        
#         out = self.sigmoid(T_NA * T_XA)
#         y_seq = out * spikes  

#         return y_seq   

 # #origin
# class DTA(nn.Module):
#     def __init__(self, T, out_channels):
#         super().__init__()

#         self.T_NA = T_NA(in_planes=out_channels*T, kernel_size=7)
#         self.T_XA = T_XA(time_step=T) 

#         self.softmax = nn.Softmax(dim=1)
#         self.sigmoid = nn.Sigmoid()
   
#     def forward(self, x_seq, spikes):
        
#         B, T, C, H, W = x_seq.shape     
        
#         x_seq_2 = x_seq.reshape(B, T*C, H, W)
#         T_NA = self.T_NA(x_seq_2) 
#         T_NA = T_NA.reshape(B, T, C, H, W)
        
#         T_XA = self.T_XA(x_seq) 
        
#         out = self.sigmoid(T_NA * T_XA)
#         y_seq = out * spikes  

#         return y_seq   