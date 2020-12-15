import torch as t 
import torch.nn as nn
import torch.nn.functional as F

class BaseLayer(nn.Module): #однослойная сеть
    def __init__(self, in_,  out_,   act=F.relu):         
        nn.Module.__init__(self)                    
        self.mean = nn.Parameter(t.randn(in_, out_, device=device)) # параметры средних
        t.nn.init.xavier_uniform(self.mean) 
        self.mean_b = nn.Parameter(t.randn(out_, device=device)) # то же самое для свободного коэффициента
                
        self.in_ = in_
        self.out_ = out_
        self.act = act
        
    def forward(self,x):    
        w = self.mean 
        b = self.mean_b
            
        # функция активации 
        return self.act(t.matmul(x, w)+b)

    def KLD(self, l):        
        # подсчет hyperloss
        return l * t.linalg.norm(self.mean)  