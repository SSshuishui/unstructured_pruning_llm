import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearMasked(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'layer' in kwargs:
            layer = kwargs['layer']
            self.mask = self.layer
        else:
            self.mask = nn.Parameter(torch.zeros(self.weight.shape))
        self.prune_rate = 0

    def forward(self, x):
        if self.prune_rate != 0:
            sparseWeight = self.mask * self.weight
            x = F.linear(
                x, sparseWeight, self.bias
            )
        else:
            x = F.linear(
                x, self.weight, self.bias
            )
        return x
