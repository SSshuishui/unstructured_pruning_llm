import torch
import torch.nn as nn

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

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
        
        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        
    def free(self):
        self.scaler_row = None
        torch.cuda.empty_cache()


# Define BiasGPT class
class FLAPGPT:
    """
    This class wraps a GPT layer for specific operations.
    """
    def __init__(self, layer, metric):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.out_dim = layer.weight.data.shape[0]
        self.in_dim = layer.weight.data.shape[1]
        self.type = metric
        self.nsamples = 0

        self.baseline_inp = torch.zeros((self.in_dim), device=self.dev)
        if self.type == "WIFN":
            self.scaler_inp = torch.zeros((self.in_dim), device=self.dev)
        else:   
            self.fluc_inp = torch.zeros((self.in_dim), device=self.dev)

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()   # (dim, seqlen)

        old_baseline_inp = self.baseline_inp
        self.baseline_inp *= self.nsamples / (self.nsamples + batch_size)
        self.baseline_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
        if self.type == "WIFN":
            inp = inp.type(torch.float32)
            self.scaler_inp *= self.nsamples / (self.nsamples + batch_size)
            self.scaler_inp += torch.norm(inp, p=2, dim=1) ** 2  / (self.nsamples + batch_size)
        else:
            if self.nsamples == 0:
                self.fluc_inp = 0
            else:
                self.fluc_inp *= (self.nsamples - 1) / (self.nsamples + batch_size - 1)
                self.fluc_inp += torch.sum((inp - self.baseline_inp.unsqueeze(1)) * (inp - old_baseline_inp.unsqueeze(1)), dim=1) / (self.nsamples + batch_size)   # a²+b²+c²...没开根号

        self.nsamples += batch_size

        
    def free(self):
        self.baseline_inp = None
        if hasattr(self, 'fluc_inp'):
            self.fluc_inp = None
        if hasattr(self, 'scaler_inp'):
            self.scaler_inp = None
        torch.cuda.empty_cache()  


"""
    'IFV': Input Feature Variance
    'WIFV': Weighted Input Feature Variance
    'WIFN': Weighted Input Feature Norm
"""
metrics = {
    'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
    'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(subset[name].weight.data.pow(2), dim=0),
    'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1)))).mean(axis=0),
}


def cal_remove_neuron(args, model):
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    if args.structure == "UL-MM":
        remove_params = args.pruning_ratio * (intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * (args.remove_heads // num_layers) * 128
        return int((remove_params - remove_head_params) / (hidden_size * 3))
    else:
        remove_params = num_layers * args.pruning_ratio * (intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * args.remove_heads * 128
        return int((remove_params - remove_head_params) / (hidden_size * 3))
    


def compress(layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, bias=False, unstr=False):
    """
    Compress a model layer by masking or pruning based on the given masks.
    
    Args:
        layer (nn.Module): The model layer to compress.
        attn_mask (torch.Tensor): The mask to apply to the attention weights.
        mlp_mask (torch.Tensor): The mask to apply to the MLP weights.
        attn_mean_inp (torch.Tensor): The mean attention input.
        mlp_mean_inp (torch.Tensor): The mean MLP input.
        device (torch.device): Device on which the model is loaded.
        bias (bool, optional): Whether to consider bias while compressing. Defaults to False.
        unstr (bool, optional): If True, only mask without real pruning. Defaults to False.
        
    Returns:
        None: This function modifies the layer in-place and doesn't return anything.
    """
    if unstr:  # Only mask, do not really prune
        # Attention Weight Masking
        if attn_mask is not None:
            retain_heads = torch.count_nonzero(attn_mask)
            attn_mask = attn_mask.repeat_interleave(128)
            # Apply the mask to the query, key and value projection weights
            layer.self_attn.q_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
            layer.self_attn.k_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
            layer.self_attn.v_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
            
            output_weight = layer.self_attn.o_proj.weight.data
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((attn_mean_inp * ~attn_mask.to(device)) @ output_weight.T)
                
            # Note: the weight data is masked, but the weight tensor shape remains unchanged
            if bias:
                layer.self_attn.o_proj.bias.data = output_bias
            layer.self_attn.o_proj.weight.data = output_weight

        # MLP Weight Masking
        if mlp_mask is not None:
            # Apply the mask to the up and gate projection weights
            layer.mlp.up_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
            layer.mlp.gate_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
            
            output_weight = layer.mlp.down_proj.weight.data
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)
                
            # Note: the weight data is masked, but the weight tensor shape remains unchanged
            if bias:
                layer.mlp.down_proj.bias.data = output_bias
            layer.mlp.down_proj.weight.data = output_weight
    
    else:
        # Real Pruning
        # Attention Weight Pruning
        if attn_mask is not None:
            retain_heads = torch.count_nonzero(attn_mask)
            attn_mask = attn_mask.repeat_interleave(128)
            
            # Prune the query, key and value projection weights
            # We reduce the size of the weights based on the attention mask
            layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(attn_mask)[0]]
            layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(attn_mask)[0]]
            layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(attn_mask)[0]]
            
            # Update output dimensions of q, k, v projections based on remaining heads
            layer.self_attn.q_proj.out_features = attn_mask.sum().item()
            layer.self_attn.k_proj.out_features = attn_mask.sum().item()
            layer.self_attn.v_proj.out_features = attn_mask.sum().item()
            
            output_weight = layer.self_attn.o_proj.weight.data
            
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((attn_mean_inp * ~attn_mask.to(device)) @ output_weight.T)
                
            # Prune the output projection weight
            output_weight = layer.self_attn.o_proj.weight.data[:, torch.where(attn_mask)[0]]
            # Update layer configurations for the new output shape after pruning
            layer.self_attn.num_heads = retain_heads
            layer.self_attn.hidden_size = retain_heads * 128
            
            if bias:
                # Re-initialize the Linear layer with new shape and bias
                layer.self_attn.o_proj.in_features = attn_mask.sum().item()
                # layer.self_attn.o_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
                layer.self_attn.o_proj.bias.data = output_bias
                
            # Assign the pruned weights
            layer.self_attn.o_proj.weight.data = output_weight

        # MLP Weight Pruning
        if mlp_mask is not None:
            # Prune the up and gate projection weights
            layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
            layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]
            
            # Update output dimensions of up and gate projections based on the mlp mask
            layer.mlp.up_proj.out_features = mlp_mask.sum().item()
            layer.mlp.gate_proj.out_features = mlp_mask.sum().item()
            
            output_weight = layer.mlp.down_proj.weight.data
            layer.mlp.intermediate_size = mlp_mask.sum().item()
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)
              
            # Prune the down projection weight
            output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]  
            
            if bias:
                # Re-initialize the Linear layer with new shape and bias
                layer.mlp.down_proj.in_features = mlp_mask.sum().item()
                # layer.mlp.down_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
                layer.mlp.down_proj.bias.data = output_bias
                
            # Assign the pruned weights
            layer.mlp.down_proj.weight.data = output_weight
        
    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()