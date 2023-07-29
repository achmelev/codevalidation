import torch
import torch.nn as nn
from torch import linalg as LA
from numbercode import NumberCode


class NumberCodeModel(nn.Module):
    
    def __init__(self, hiddenLayerOrder = 3):
        super().__init__()
        self.origin = None
        hiddenLayerSize = 2**hiddenLayerOrder
        self.network = nn.Sequential()
        ## Input Layer
        self.network.append(nn.Linear(NumberCode.codeWith*10,hiddenLayerSize))
        self.network.append(nn.ReLU())

        while hiddenLayerSize > 4:
            self.network.append(nn.Linear(hiddenLayerSize, int(hiddenLayerSize/2)))
            self.network.append(nn.ReLU())
            hiddenLayerSize = int(hiddenLayerSize/2)
        
        #Output Layer
        self.network.append(nn.Linear(hiddenLayerSize, 2))

    def forward(self, x):
        return self.network(x)
    
    def all_params(self, from_origin = False):
        if (from_origin):
            if (self.origin is None):
                raise Exception("origin not initialized")
            else:
                current_state = self.all_params()
                return current_state.add(self.origin,alpha=-1.0)
        else:
            params = self.parameters()
            tensorlist = []
            for p in params:
                tensorlist.append(p.data.view(-1))
            return torch.cat(tensorlist, 0)
    
    def all_grads(self):
        params = self.parameters()
        tensorlist = []
        for p in params:
            tensorlist.append(p.grad.view(-1))
        return tensorlist
    
    def params_norm(self, from_origin = True):
        return LA.norm(self.all_params(from_origin)).item()
    
    def grads_norm(self):
        result = []
        tensor_list = self.all_grads()
        for tensor in tensor_list:
            result.append(LA.norm(tensor).item())
        result.append(LA.norm(torch.cat(tensor_list)).item())
        return result
    
    def init_origin(self):
        self.origin = self.all_params()


