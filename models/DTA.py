from models.TXA import T_XA
from models.TNA import T_NA
import torch.nn as nn

class DTA(nn.Module):
    def __init__(self, T, out_channels):
        super().__init__()

        self.T_NA = T_NA(in_planes=out_channels*T, kernel_size=7)
        self.T_XA = T_XA(time_step=T) 

        self.router = nn.Linear(out_channels * T, 2)  # 두 expert의 가중치 결정
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq, spikes):
        
        B, T, C, H, W = x_seq.shape 

        #for moe
        # Router의 가중치 결정
        x_flat = x_seq.view(B, -1)  # B x (T * C * H * W) <--- 의미 없는 작업같음
        expert_weights = self.softmax(self.router(x_flat))  # B x 2
        
        x_seq_2 = x_seq.reshape(B, T*C, H, W)
        T_NA = self.T_NA(x_seq_2) 
        T_NA = T_NA.reshape(B, T, C, H, W)
        
        T_XA = self.T_XA(x_seq) 

        out = expert_weights[:, 0].view(B, 1, 1, 1, 1) * T_NA + \
                    expert_weights[:, 1].view(B, 1, 1, 1, 1) * T_XA #덧셈 곱셈 실험
        
        #mixed_output_sigmoid = self.sigmoid(mixed_output)

        y_seq = out * spikes  

        return y_seq  
        
    def forward(self, x_seq, spikes):
        
        B, T, C, H, W = x_seq.shape 
        
        x_seq_2 = x_seq.reshape(B, T*C, H, W)
        T_NA = self.T_NA(x_seq_2) 
        T_NA = T_NA.reshape(B, T, C, H, W)
        
        T_XA = self.T_XA(x_seq) 
        
        out = self.sigmoid(T_NA * T_XA)
        y_seq = out * spikes  

        return y_seq   