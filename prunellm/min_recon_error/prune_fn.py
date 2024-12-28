import torch
import torch.nn as nn 
from .linear_type import LinearMasked

def find_layers(module, layers=[nn.Linear], masked_layers=[LinearMasked], name=''):
    """
    Recursively find the layers of a certain type in a module.

    config:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers or type(module) in masked_layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, masked_layers=masked_layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def prune_magnitude(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    W = layer.weight.data 
    M = layer.mask.data
    W_metric = torch.abs(W)
    if prune_n != 0:
        W_mask = (torch.zeros_like(W)==1)
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:,ii:(ii+prune_m)].float()
                W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=True)[1], True)
    else:
        thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*sparsity_ratio)].cpu()
        W_mask = (W_metric>thresh)

    M[W_mask] = 1

def prune_wanda(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    W_metric = torch.abs(layer.weight.data) * torch.sqrt(wrapped_layer.scaler_row.reshape((1,-1)))

    if prune_n != 0:
        # structured n:m sparsity
        final_indices = None
        with torch.no_grad():
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    indices = ii+torch.topk(tmp, prune_n,dim=1, largest=True)[1].cuda()
                    layer.mask.scatter_(1, indices, 1)
                    if final_indices is None:
                        final_indices = indices.cpu()
                    else:
                        final_indices = torch.concat((final_indices,indices.cpu()),1)
    else:
        sort_res = torch.sort(W_metric, dim=-1, stable=True, descending=True)

        #unstructured pruning
        indices = sort_res[1][:,:int(W_metric.shape[1]*(1 - sparsity_ratio))]
        with torch.no_grad():
            layer.mask.scatter_(1, indices.cuda(), 1)

def prune_sparsegpt(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    wrapped_layer.fasterprune(sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
    wrapped_layer.free()


def obtain_output(sub_layers, inp, attention_mask, position_ids, device=torch.device("cuda:0"), offload=False):
    """
    Obtain outputs for inp when passing through sub_layers
    """
    for sub_layer_idx in range(len(sub_layers)):
        if offload:
            sub_layers[sub_layer_idx] = sub_layers[sub_layer_idx].to(device)
        if sub_layer_idx == 0:
            out = sub_layers[sub_layer_idx](inp, attention_mask=attention_mask, position_ids=position_ids)[0]
        else:
            out = sub_layers[sub_layer_idx](out, attention_mask=attention_mask, position_ids=position_ids)[0]
        
        if offload:
            sub_layers[sub_layer_idx] = sub_layers[sub_layer_idx].to('cpu')
    return out