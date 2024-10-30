import math

import torch
import torch.nn as nn
import transformers


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def return_reorder_indice(input_tensor):
    """
    For instance:
    [[1., -2., 3.],
    [-2, 2., -4],
    [5., 6., -7],
    [-6, -7, -4]]
    return indices of
    [[-2.,  3.,  1.],
    [-2., -4.,  2.],
    [-7.,  6.,  5.],
    [-6., -7., -4.]]
    Description: The relative order in the positive number remains unchanged, and the relative order in the negative number is flipped.
    """
    positive_tensor = input_tensor.clone()
    negative_tensor = input_tensor.clone()

    positive_mask = positive_tensor > 0
    negative_mask = negative_tensor < 0

    positive_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )
    negative_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )

    positive_indices[~positive_mask] = float("inf")
    negative_indices[~negative_mask] = float("inf")

    positive_value, _ = torch.sort(positive_indices, dim=1)
    negative_value, _ = torch.sort(negative_indices, dim=1)

    positive_value = torch.flip(positive_value, dims=[1])

    negative_value[negative_value == float("inf")] = 0
    positive_value[positive_value == float("inf")] = 0

    reorder_indice = (positive_value + negative_value).to(torch.int64)

    return reorder_indice
    

# Define WrappedGPT class
class DSnoTGPT:
    """
    This class wraps a GPT layer for specific operations.
    """
    def __init__(self, layer, initial_method = None, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.nsamples = 0

        self.initial_method = initial_method
        if self.initial_method == "sparsegpt":
            self.H = torch.zeros((self.columns, self.columns), device=self.dev)

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.sum_metric_row = torch.zeros((self.columns), device=self.dev)
        
        self.mean = torch.zeros((self.columns), device=self.dev)
        self.var = torch.zeros((self.columns), device=self.dev)
        self.ntokens = 0
        
        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        inp = inp.type(torch.float32)

        mean_inp = torch.mean(inp, dim=1, keepdim=True)

        var_inp = torch.var(inp, dim=1, unbiased=False, keepdim=True)
        num_inp = inp.shape[1]
        self.var = var_inp if self.ntokens == 0 else (self.var * self.ntokens + var_inp * num_inp) / (self.ntokens + num_inp)
        self.mean = mean_inp if self.ntokens == 0 else (self.mean * self.ntokens + mean_inp * num_inp) / (self.ntokens + num_inp)
        self.ntokens += num_inp
        
        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.sum_metric_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        self.sum_metric_row += torch.sum(inp, dim=1) / self.nsamples

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
