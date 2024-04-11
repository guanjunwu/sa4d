import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import sys

class SegNet(nn.Module):
    def __init__(self, args):
        super(SegNet, self).__init__()
        timebase_pe = args.timebase_pe
        posbase_pe= args.posebase_pe
        self.input_ch = (3 + (3 * posbase_pe) * 2) + (1 + (1 * timebase_pe) * 2)
        self.output_ch = 32
        self.W = 256
        self.D = 4
        self.mlp = nn.ModuleList(
            [nn.ReLU(), nn.Linear(self.input_ch, self.W)] +
            sum([[nn.ReLU(), nn.Linear(self.W, self.W)] for i in range(self.D-2)], []) +
            [nn.ReLU(), nn.Linear(self.W, self.output_ch)] 
            # [nn.Sigmoid()]
        )
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.apply(initialize_weights)
           
    def forward(self, point, time):
        point_emb = poc_fre(point, self.pos_poc)
        time_emb = poc_fre(time, self.time_poc)
        # time_emb = time
        h = torch.cat([point_emb, time_emb], -1)
        for i, l in enumerate(self.mlp):
            h = self.mlp[i](h)
        
        return h

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)

def poc_fre(input_data, poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb