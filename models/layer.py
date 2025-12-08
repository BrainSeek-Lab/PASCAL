from cv2 import mean
from sympy import print_rcode
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import numpy as np

spike_rate = [0 for i in range(30)]

torch.set_printoptions(precision=20)
class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= -8e-5).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class GradFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

myfloor = GradFloor.apply

class IF(nn.Module):
    def __init__(self, index, mylist, T=0, L=8, thresh=1.0, tau=1., gama=1.0):
        super(IF, self).__init__()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)
        self.tau = tau
        self.gama = gama
        self.loss = 0
        self.image_id = 0
        self.count_tensor = torch.tensor(0).reshape(1)
        self.index = index
        self.list = mylist
        self.L = self.list[self.index]
        self.T = self.list[self.index]
        if(self.index > 0):
            self.expand = ExpandTemporalDim(self.list[self.index-1])
        else:
            self.expand =  ExpandTemporalDim(self.list[self.index])
        self.merge = MergeTemporalDim(self.list[self.index])

    def forward(self, x):
        if self.T > 0:
            if(self.index == 0): 
                x = x / self.thresh
                x = torch.clamp(x, 0, 1)
                x = myfloor(x*self.L+0.5)/self.L
                x = x * self.thresh
                thre = self.thresh.data/self.L
                spike_pot = []
                for t in range(self.L):
                    spike = self.act(x-thre,self.gama)*thre
                    x = x - spike
                    spike_pot.append(spike)
                    
                x = torch.stack(spike_pot, dim = 0) 
                spike_rate[self.index] = spike_rate[self.index] + torch.mean(x/thre)*self.L
                x = self.merge(x)

                return x       
            else:
                thre = self.thresh.data/self.L
                mem = 0.5*thre
                x = self.expand(x) 
                x1 = x.mean(0)*self.list[self.index-1]
                x1 = x1 / self.thresh
                x1 = torch.clamp(x1, 0, 1)
                x1 = myfloor(x1*self.L+0.5)/self.L
                x1 = x1 * self.thresh
                
                spike_count = torch.zeros(x[0].shape, device=0)
                T1 = self.list[self.index - 1]
                if(thre > 0):
                    for t in range(T1):
                        mem = mem + x[t, ...]
                        spike = self.act(mem-thre, self.gama) * thre
                        mem = mem - spike
                        spike_count = spike_count + spike
                    T2 = self.list[self.index]
                    T3 = max(T1,T2)
                    for t in range(T3-1):
                        spike = self.act(mem-thre,self.gama)*thre
                        rev_spike = ((-mem)>0).float()*thre #no reverse spike if mem = 0
                        mem = mem - spike + rev_spike
                        spike_count = spike_count + spike - rev_spike
                    mem = spike_count
                    spike_pot = []
                    for t in range(T2):
                        spike = self.act(mem-thre,self.gama) * thre
                        mem = mem - spike
                        spike_pot.append(spike)
                else:
                    for t in range(T1):
                        mem = mem + x[t, ...]
                        spike = self.act(thre-mem, self.gama) * thre
                        mem = mem - spike
                        spike_count = spike_count + spike
                    T2 = self.list[self.index]
                    T3 = max(T1,T2)
                    for t in range(T3-1):
                        spike = self.act(thre-mem,self.gama)*thre
                        rev_spike = ((mem)>0).float()*thre #no reverse spike if mem = 0
                        mem = mem - spike + rev_spike
                        spike_count = spike_count + spike - rev_spike
                    mem = spike_count
                    spike_pot = []
                    for t in range(T2):
                        spike = self.act(thre-mem,self.gama) * thre
                        mem = mem - spike
                        spike_pot.append(spike)
                x = torch.stack(spike_pot, dim = 0)   
                x = self.merge(x)
                return x  
        else:
            x = x / self.thresh
            x = torch.clamp(x, 0, 1)
            x = myfloor(x*self.L+0.5)/self.L
            x = x * self.thresh

            return x
def add_dimention(x, T):
    x.unsqueeze_(1) 
    x = x.repeat(T, 1, 1, 1, 1)
    return x
