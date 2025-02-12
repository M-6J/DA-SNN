import torch.nn as nn
import torch 

class T_XA(nn.Module):
    def __init__(self, time_step):
        super(T_XA, self).__init__()
        self.conv_t = nn.Conv1d(in_channels=time_step, out_channels=time_step, 
                              kernel_size=4, padding='same', bias=False)
        
        self.conv_c = nn.Conv1d(in_channels=32, out_channels=32,
                                kernel_size=4, padding='same', bias=False)

        self.sigmoid = nn.Sigmoid()
        
        self.scale_t = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.scale_c = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        
    def forward(self, x_seq):     #input BTCHW
        x_t = torch.mean(x_seq.permute(0, 1, 2, 3, 4), dim=[3, 4]) 
        x_c = x_t.permute(0, 2, 1) 

        conv_t_out = self.conv_t(x_t)
        attn_map_t = self.sigmoid(conv_t_out)

        conv_c_out = self.conv_c(x_c).permute(0, 2, 1)  
        attn_map_c = self.sigmoid(conv_c_out)
        
        after_scale_t = attn_map_t * self.scale_t 
        after_scale_c = attn_map_c*self.scale_c
        
        attn_ft = after_scale_t[:, :, :, None, None] * after_scale_c[:, :, :, None, None]
        y_seq = x_seq * attn_ft

        return y_seq
        
    # def forward(self, x_seq):    #input TNCHW 
    #     x_t = torch.mean(x_seq.permute(1, 0, 2, 3, 4), dim=[3, 4]) #NTC
    #     x_c = x_t.permute(0, 2, 1) #NCT

    #     conv_t_out = self.conv_t(x_t).permute(1, 0, 2) #NTC -> TNC
    #     attn_map_t = self.sigmoid(conv_t_out)

    #     conv_c_out = self.conv_c(x_c).permute(2, 0, 1) #NCT -> TNC
    #     attn_map_c = self.sigmoid(conv_c_out)
        
    #     after_scale_t = attn_map_t * self.scale_t 
    #     after_scale_c = attn_map_c*self.scale_c
    #     # dta origin
    #     # attn_t_ft = x_seq + after_scale_t[:, :, :, None, None]
    #     # attn_c_ft = x_seq + after_scale_c[:, :, :, None, None]

    #     # y_seq = attn_t_ft * attn_c_ft 

    #     #gpt
    #     attn_ft = after_scale_t[:, :, :, None, None] * after_scale_c[:, :, :, None, None]
    #     y_seq = x_seq * attn_ft

    #     return y_seq


# #origin TCJA
# class T_XA(nn.Module):
#     def __init__(self, time_step):
#         super(T_XA, self).__init__()
#         self.conv_t = nn.Conv1d(in_channels=time_step, out_channels=time_step, 
#                               kernel_size=4, padding='same', bias=False)
        
#         self.conv_c = nn.Conv1d(in_channels=32, out_channels=32,
#                                 kernel_size=4, padding='same', bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x_seq: torch.Tensor):
#         x = torch.mean(x_seq.permute(1, 0, 2, 3, 4), dim=[3, 4])
#         x_c = x.permute(0, 2, 1)
#         conv_t_out = self.conv_t(x).permute(1, 0, 2)
#         conv_c_out = self.conv_c(x_c).permute(2, 0, 1)
#         out = self.sigmoid(conv_c_out * conv_t_out)
#         y_seq = x_seq * out[:, :, :, None, None]
#         return y_seq

# #DTA origin 
# class T_XA(nn.Module):
#     def __init__(self, time_step):
#         super(T_XA, self).__init__()
#         self.conv_t = nn.Conv1d(in_channels=time_step, out_channels=time_step, 
#                               kernel_size=3, padding='same', bias=False)
        
#         self.conv_c = nn.Conv1d(in_channels=64, out_channels=64,
#                                 kernel_size=6, padding='same', bias=False)

#         self.sigmoid = nn.Sigmoid()
        
#         self.scale_t = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
#         self.scale_c = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        
#     def forward(self, x_seq):    
#         x_t = torch.mean(x_seq.permute(0, 1, 2, 3, 4), dim=[3, 4]) 
#         x_c = x_t.permute(0, 2, 1) 

#         conv_t_out = self.conv_t(x_t)
#         attn_map_t = self.sigmoid(conv_t_out)

#         conv_c_out = self.conv_c(x_c).permute(0, 2, 1)  
#         attn_map_c = self.sigmoid(conv_c_out)
        
#         after_scale_t = attn_map_t * self.scale_t 
#         after_scale_c = attn_map_c*self.scale_c
        
#         attn_t_ft = x_seq + after_scale_t[:, :, :, None, None]
#         attn_c_ft = x_seq + after_scale_c[:, :, :, None, None]

#         y_seq = attn_t_ft * attn_c_ft 

#         return y_seq
