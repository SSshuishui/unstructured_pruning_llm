import time
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import numpy as np
import math
import transformers

from utils.logutils import create_logger
from utils.datautils import get_loaders
from utils.modelutils import *


def get_llama(model):    
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, device_map='auto', torch_dtype='auto')
    model.seqlen = 2048
    return model


def llama_sequential_magnitude(args, model, dataloader, dev, logger, ratios=None):
    logger.info("Starting...")

    layers = model.model.layers 

    hf_device_map = model.hf_device_map

    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(len(layers))]

    k = 0
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if args.prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % args.prune_m == 0:
                        tmp = W_metric[:,ii:(ii+args.prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, args.prune_n, dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*ratios[k])].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

    torch.cuda.empty_cache()


@torch.no_grad()
def llama_sequential_sparsegpt(args, model, dataloader, dev, logger, ratios=None):
    logger.info("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    logger.info("Ready.")

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(len(layers))]

    quantizers = {}
    k=0
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gpts = {}
            for name in subset:
                if (
                    not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
                ) == (not args.invert):
                    continue
                gpts[name] = SparseGPT(subset[name])
                if args.wbits < 16:
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=False, mse=False
                    )

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                logger.info("Pruning ...")
                gpts[name].fasterprune(
                    ratios[k],
                    prunen=args.prune_n,
                    prunem=args.prune_m,
                    percdamp=args.percdamp
                )
                gpts[name].free()
                k += 1

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def llama_sequential_wanda(args, model, dataloader, dev, logger, ratios=None):
    logger.info("Starting...")

    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache['position_ids']

    logger.info('Ready.')

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(len(layers))]
    k=0

    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = find_layers(layer)

        wanda_layers = {}
        for name in subset:
            wanda_layers[name] = WandaGPT(subset[name])
        def add_batch(name):
            def tmp(_, inp, out):
                wanda_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(i, name)
            logger.info("Pruning ...")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wanda_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if args.prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % args.prune_m == 0:
                        tmp = W_metric[:,ii:(ii+args.prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, args.prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    logger.info(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*ratios[k])]
                    k += 1
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def llama_sequential_sparsellm(args, model, dataloader, dev, logger):
    logger.info("Starting...")

    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache['position_ids']

    logger.info('Ready.')

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        full = find_layers(layer)
        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gpts = {}
            for name in subset:
                if (
                    not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
                ) == (not args.invert):
                    continue
                gpts[name] = SparseLLM(subset[name])
                if args.wbits < 16:
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=False, mse=False
                    )
            
            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data, name)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            target_layer_names = ["mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"]

            for name in subset:
                if name not in target_layer_names:
                    print(i, name)
                    logger.info("Pruning ...")
                    gpts[name].fasterprune(
                        args.sparsity_ratio,
                        prunen=args.prune_n,
                        prunem=args.prune_m,
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
                    )
                    gpts[name].free()

            # Adjust hyperparameters as needed
            alpha = 5.0  # 对应全局中的 alpha
            beta = 5.0  # 对应激活更新， zl更新中的 alpha
            gamma = 5.0 # 对应权重中的beta，激活更新，zl更新中的 beta

            # Define the number of global pruning epochs
            opt_epochs = 8  # This might need to be adjusted

            # Get the inputs and outputs which are constants here
            X_list = gpts['mlp.up_proj'].batch_inp
            Y_list = gpts['mlp.down_proj'].batch_out
            X = torch.stack(X_list, dim=0)
            Y = torch.stack(Y_list, dim=0)
            # Reshape to 2D
            X, Y = X.reshape((-1, X.size(-1))).T, Y.reshape((-1, Y.size(-1))).T

            # free memory 
            X_list, Y_list = None, None
            gpts['mlp.up_proj'].batch_inp.clear()
            gpts['mlp.down_proj'].batch_out.clear()

            # Get the hidden variables and their initialization
            # z: output of 'mlp.up_proj'  对应论文中的z_l
            hidden_z_list = gpts['mlp.up_proj'].batch_out
            z = torch.stack(hidden_z_list, dim=0)
            hidden_z_list = None
            gpts['mlp.up_proj'].batch_out.clear()
            # p: input of 'mlp.down_proj'  对应论文中的 a_l
            hidden_p_list = gpts['mlp.down_proj'].batch_inp
            p = torch.stack(hidden_p_list, dim=0)
            hidden_p_list = None
            gpts['mlp.down_proj'].batch_inp.clear()
            # s: output of 'mlp.gate_proj'  对应论文中的 s_l
            hidden_s_list = gpts['mlp.gate_proj'].batch_out
            s = torch.stack(hidden_s_list, dim=0)
            hidden_s_list = None
            gpts['mlp.gate_proj'].batch_out.clear()

            # Reshape auxiliary variables
            z = z.reshape((-1, z.size(-1))).T.to(dev)
            p = p.reshape((-1, p.size(-1))).T.to(dev)
            s = s.reshape((-1, s.size(-1))).T.to(dev)

            torch.cuda.empty_cache()

            # Pre-compute the pinverse of X and cache it to save computational cost
            Xinv = torch.pinverse(X.to(dtype=torch.float32)).half()

            # list to store training losses
            training_loss = {'Y_p_loss': [], 'p_z_loss': [], 'z_X_loss': [], 'train_loss': []}

            for opt_step in range(opt_epochs):
                ##############
                # optimize W
                ##############
                if opt_step > 0:   # for the first step, no need for updating W
                    # Update the weight matrix of mlp.up_project
                    # Calculate the weight matrix
                    weight_matrix_1 = torch.matmul(z, Xinv)
                    # assign the new parameters to gpts class
                    gpts['mlp.up_proj'].layer.weight.copy_(weight_matrix_1)
                    del weight_matrix_1

                    # Update the weight matrix of mlp.down_proj
                    pinv = torch.pinverse(p.to(dtype=torch.float32)).half()
                    # Calculate the weight matrix
                    weight_matrix_2 = torch.matmul(Y, pinv)
                    # assign the new parameters to gpts class
                    gpts['mlp.down_proj'].layer.weight.copy_(weight_matrix_2)
                    del weight_matrix_2, pinv

                    # Update the weight matrix of mlp.gate_project
                    # Calculate the weight matrix
                    weight_matrix_3 = torch.matmul(s, Xinv)
                    # assign the new parameters to gpts class
                    gpts['mlp.gate_proj'].layer.weight.copy_(weight_matrix_3)
                    del weight_matrix_3

                    torch.cuda.empty_cache()

                ##############
                # prune W
                ##############
                # modify gpts[name].H to be our auxiliary variable
                if opt_step > 0:   # for the first step, no need for updating H    
                    tmp_H = torch.zeros_like(gpts['mlp.down_proj'].H)
                    tmp_p = p.T.reshape((args.nsamples, -1, p.size(0)))
                    tmp_nsamples = 0
                    for j in range(args.nsamples):
                        tmp_inp = tmp_p[j].unsqueeze(0)
                        tmp = tmp_inp.shape[0]
                        if isinstance(gpts['mlp.down_proj'].layer, nn.Linear) or isinstance(gpts['mlp.down_proj'].layer, transformers.Conv1D):
                            if len(tmp_inp.shape) == 3:
                                tmp_inp = tmp_inp.reshape((-1, tmp_inp.shape[-1]))
                            tmp_inp = tmp_inp.t()
                        tmp_H *= tmp_nsamples / (tmp_nsamples + tmp)
                        tmp_nsamples += tmp
                        tmp_inp = math.sqrt(2 / tmp_nsamples) * tmp_inp.float()
                        tmp_H += tmp_inp.matmul(tmp_inp.t())
                    gpts['mlp.down_proj'].H.copy_(tmp_H)
                    del tmp_H, tmp_p
                    torch.cuda.empty_cache()

                for name in target_layer_names:
                    print(i, name)
                    print('FFN Pruning ...')
                    gpts[name].fasterprune(
                        args.sparsity_ratio,
                        prunen=args.prune_n,
                        prunem=args.prune_m,
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
                    )
                
                ##############
                # optimize p
                ##############
                logger.info('Optimizing p ...')
                # Activation inverse
                # beta 对应论文中的alpha
                # gamma 对应论文中的beta
                next_weight = subset['mlp.down_proj'].weight
                m1 = beta * torch.matmul(next_weight.T, next_weight)
                m2 = gamma * torch.eye(m1.shape[0], device=m1.device)
                av = torch.inverse(m1 + m2).to(dtype=torch.float16)

                del m1, m2
                torch.cuda.empty_cache()

                # Calculate SwiGLU output
                layer_nl_output = nn.functional.silu(s) * z

                # Activation formulate
                m3 = beta * torch.matmul(next_weight.T, Y)
                m4 = gamma * layer_nl_output
                af = m3 + m4

                p = torch.matmul(av, af)

                del layer_nl_output, next_weight, av, m3, m4, af
                torch.cuda.empty_cache()

                ##############
                # optimize z
                ##############
                logger.info('Optimizing z ...')
                w = subset['mlp.up_proj'].weight
                m = torch.matmul(w, X)
                swish = nn.functional.silu(s)
                z = (m + swish * p) / (swish ** 2 + 1)    

                del w, m, swish
                torch.cuda.empty_cache()
                
                ##############
                # optimize s
                ##############
                logger.info('Optimizing s ...')
                w = subset['mlp.gate_proj'].weight
                # convert the layer's weight tensor to float32 and enable grad
                w = w.to(dtype=torch.float32).requires_grad_(True)

                s_update_epochs = 2
                s_learning_rate = 0.01
                for _ in range(s_update_epochs):

                    batch_size = 1000  # Choose an appropriate batch size based on your memory constraints
                    # s: [hidden_d, n_samples]
                    for k in range(0, s.size(-1), batch_size):
                        chunk = slice(k, k + batch_size)

                        # get the "mini-batch" for each tensor and turn on autograd
                        X_batch = X[:,chunk].to(dtype=torch.float32).requires_grad_(True)
                        z_batch = z[:,chunk].to(dtype=torch.float32).requires_grad_(True)
                        p_batch = p[:,chunk].to(dtype=torch.float32).requires_grad_(True)
                        s_batch = s[:,chunk].to(dtype=torch.float32).requires_grad_(True)

                        with torch.enable_grad():   # temporarily turn on the Pytorch computational graph functionality
                            loss_s = alpha * torch.norm(s_batch - torch.matmul(w, X_batch))**2
                            loss_s += gamma * torch.norm(p_batch - nn.functional.silu(s_batch) * z_batch)**2

                        loss_s.backward()
                        s_batch -= s_learning_rate * s_batch.grad  
                        s_batch.grad.zero_()
                        s[:,chunk] = s_batch.detach().to(dtype=torch.float16)

                s_batch, X_batch, z_batch, p_batch, w = s_batch.detach(), X_batch.detach(), z_batch.detach(), p_batch.detach(), w.detach()
                del w, loss_s, s_batch, X_batch, z_batch, p_batch
                torch.cuda.empty_cache()

                # compute and save the training loss after each epoch
                tmp_training_loss = nn.functional.mse_loss(torch.matmul(subset['mlp.down_proj'].weight, 
                                                                        nn.functional.silu(torch.matmul(subset['mlp.gate_proj'].weight, X)) 
                                                                        * torch.matmul(subset['mlp.up_proj'].weight, X)), Y)
                training_loss['train_loss'].append(tmp_training_loss.item())

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def llama_sequential_DSnoT(args, model, dataloader, dev, logger):
    logger.info("Starting...")
    
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache['position_ids']

    logger.info('Ready.')

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = DSnoTGPT(
                subset[name],
                initial_method=args.initial_method
            )
        
        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")

            DSnoT_metric = subset[name].weight.data * wrapped_layers[name].sum_metric_row.reshape((1, -1))

            if args.initial_method == "wanda":
                initial_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1))
                )
            elif args.initial_method == "magnitude":
                initial_metric = torch.abs(subset[name].weight.data)
            elif args.initial_method == "sparsegpt":
                W = subset[name].weight.data.clone()
                if isinstance(subset[name], nn.Conv2d):
                    W = W.flatten(1)
                if isinstance(subset[name], transformers.Conv1D):
                    W = W.t()
                W = W.float()

                H = wrapped_layers[name].H
                # del wrapped_layers[name].H
                dead = torch.diag(H) == 0
                H[dead, dead] = 1
                W[:, dead] = 0

                percdamp = 0.01
                damp = percdamp * torch.mean(torch.diag(H))
                diag = torch.arange(
                    wrapped_layers[name].columns, device=wrapped_layers[name].dev
                )
                H[diag, diag] += damp
                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                Hinv = H

                initial_metric = W**2 / (torch.diag(Hinv).reshape((1, -1))) ** 2
            
            weight_mask = torch.zeros_like(initial_metric) == 1

            if args.prune_n != 0:
                if (name.split(".")[0] == args.skip_layer or name.split(".")[1] == args.skip_sub_layer):
                    for ii in range(initial_metric.shape[1]):
                        if ii % args.prune_m == 0:
                            tmp = initial_metric[:, ii : (ii + args.prune_m)].float()
                            weight_mask.scatter_(1, ii + torch.topk(tmp, args.prune_n, dim=1, largest=False)[1], True,)
                else:
                    initial_prune_indices = torch.zeros((initial_metric.shape[0], 0), dtype=torch.int64, device=initial_metric.device,)
                    initial_res_indices = torch.zeros((initial_metric.shape[0], 0), dtype=torch.int64, device=initial_metric.device,)

                    for ii in range(initial_metric.shape[1]):
                        if ii % args.prune_m == 0:
                            tmp = initial_metric[:, ii : (ii + args.prune_m)].float()
                            _, tmp_all_indices = torch.sort(tmp, dim=1)
                            tmp_all_indices += ii
                            res_prune_n = args.prune_m - args.prune_n
                            tmp_indices, tmp_res_indices = torch.split(
                                tmp_all_indices,
                                split_size_or_sections=[args.prune_n, res_prune_n],
                                dim=1,
                            )

                            initial_prune_indices = torch.cat(
                                (initial_prune_indices, tmp_indices), dim=1
                            )
                            initial_res_indices = torch.cat(
                                (initial_res_indices, tmp_res_indices), dim=1
                            )
                            weight_mask.scatter_(1, tmp_indices, True)

                    metric_for_regrowing = DSnoT_metric.clone()

                    metric_for_regrowing.scatter_(1, initial_res_indices, 0)

                    reconstruction_error = torch.sum(metric_for_regrowing, dim=1, keepdim=True)
                    initialize_error_sign = torch.sign(reconstruction_error)

                    if args.pow_of_var_regrowing:
                        metric_for_regrowing /= torch.pow(
                            wrapped_layers[name].var.reshape((1, -1)),
                            args.pow_of_var_regrowing,
                        )

                    _, regrowing_indices_block = torch.sort(metric_for_regrowing, dim=1, stable=True)

                    indice_indice_list_for_regrowing = torch.zeros(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    last_one = regrowing_indices_block.shape[-1] - 1
                    indice_indice_list_for_regrowing[:, 1] = last_one
                    update_num_for_regrowing = torch.ones(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    update_num_for_regrowing[:, 1] = -1

                    initial_metric.scatter_(1, initial_prune_indices, float("inf"))
                    W_metric_max_value = (torch.max(initial_metric, dim=1, keepdim=True)[0] + 1)

                    cycle_time = 1
                    update_mask = torch.ones_like(
                        reconstruction_error, dtype=torch.bool
                    )
                    while not (
                        torch.all(update_mask == False)
                        or cycle_time > args.max_cycle_time
                    ):
                        cycle_time += 1

                        # regrowing
                        indice_of_indice_indice_list_for_regrowing = (
                            (reconstruction_error > 0).int().to(torch.int64)
                        )
                        indice_indice_for_regrowing = torch.gather(
                            indice_indice_list_for_regrowing,
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                        )

                        regrowing_indice = torch.gather(
                            regrowing_indices_block,
                            1,
                            indice_indice_for_regrowing.to(torch.int64),
                        )

                        regrowing_metric = DSnoT_metric.gather(
                            1, regrowing_indice.to(torch.int64)
                        )

                        recover_block_start_indice = (
                            regrowing_indice - regrowing_indice % args.prune_m
                        )

                        recover_block_indices = (
                            torch.arange(
                                0, args.prune_m, device=recover_block_start_indice.device
                            ).repeat(recover_block_start_indice.shape[1], 1)
                            + recover_block_start_indice
                        )

                        pruning_block = torch.gather(
                            initial_metric, 1, recover_block_indices.to(torch.int64)
                        )

                        pruning_wanda_metric, pruning_indice = torch.topk(
                            pruning_block, 1, dim=1, largest=False
                        )

                        pruning_indice += recover_block_start_indice

                        
                        pruning_metric = DSnoT_metric.gather( 1, pruning_indice.to(torch.int64) )
                        

                        reconstruction_error_after = ( reconstruction_error + pruning_metric - regrowing_metric )

                        update_mask = (update_mask & ( initialize_error_sign == torch.sign(reconstruction_error_after) ) & ( abs(reconstruction_error) > args.update_threshold))

                        initial_metric.scatter_(1, pruning_indice, W_metric_max_value)

                        weight_mask.scatter_(1, pruning_indice, update_mask)

                        weight_mask.scatter_(1, regrowing_indice, ~update_mask)

                        reconstruction_error += torch.where(
                            update_mask,
                            pruning_metric,
                            torch.zeros_like(pruning_metric),
                        )
                        reconstruction_error -= torch.where(
                            update_mask,
                            regrowing_metric,
                            torch.zeros_like(regrowing_metric),
                        )

                        indice_indice_list_for_regrowing.scatter_(
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                            indice_indice_for_regrowing
                            + update_num_for_regrowing.gather(
                                1, indice_of_indice_indice_list_for_regrowing
                            ),
                        )
            else:
                _, sorted_initial_indice = torch.sort(
                    initial_metric, dim=-1, stable=True
                )

                sparsity_num = int(initial_metric.shape[1] * args.sparsity_ratio)
                res_sparsity_num = sorted_initial_indice.shape[1] - sparsity_num

                initial_prune_indices, initial_res_indices = torch.split(
                    sorted_initial_indice,
                    split_size_or_sections=[sparsity_num, res_sparsity_num],
                    dim=1,
                )

                if (
                    name.split(".")[0] == args.skip_layer
                    or name.split(".")[1] == args.skip_sub_layer
                    or args.without_DSnoT
                ):
                    weight_mask.scatter_(1, initial_prune_indices, True)

                else:
                    weight_mask.scatter_(1, initial_prune_indices, True)

                    metric_for_regrowing = DSnoT_metric.clone()
                    wanda_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                        wrapped_layers[name].scaler_row.reshape((1, -1))
                    )

                    metric_for_regrowing.scatter_(1, initial_res_indices, 0)
                    reconstruction_error = torch.sum(
                        metric_for_regrowing, dim=1, keepdim=True
                    )
                    initialize_error_sign = torch.sign(reconstruction_error)

                    if args.pow_of_var_regrowing:
                        metric_for_regrowing /= torch.pow(
                            wrapped_layers[name].var.reshape((1, -1)),
                            args.pow_of_var_regrowing,
                        )

                    _, regrowing_indices_block = torch.sort(
                        metric_for_regrowing, dim=1, stable=True
                    )

                    wanda_metric.scatter_(1, initial_prune_indices, float("inf"))
                    wanda_res_indices, _ = torch.split(
                        torch.sort(wanda_metric, dim=1, stable=True)[1],
                        split_size_or_sections=[res_sparsity_num, sparsity_num],
                        dim=1,
                    )
                    reorder_indice_of_pruning_indice = return_reorder_indice(
                        torch.gather(DSnoT_metric, 1, wanda_res_indices)
                    )
                    pruning_indices_block = torch.gather(
                        wanda_res_indices, 1, reorder_indice_of_pruning_indice
                    )

                    indice_indice_list_for_regrowing = torch.zeros(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    last_one = regrowing_indices_block.shape[-1] - 1
                    indice_indice_list_for_regrowing[:, 1] = last_one

                    update_num_for_regrowing = torch.ones(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    update_num_for_regrowing[:, 1] = -1

                    indice_indice_list_for_pruning = torch.zeros(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    last_one = pruning_indices_block.shape[-1] - 1
                    indice_indice_list_for_pruning[:, 1] = last_one

                    update_num_for_pruning = torch.ones(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    update_num_for_pruning[:, 1] = -1

                    update_mask = torch.ones_like(
                        reconstruction_error, dtype=torch.bool
                    )
                    cycle_time = 0
                    while not ( torch.all(update_mask == False) or cycle_time >= args.max_cycle_time ):
                        cycle_time += 1
                        
                        # regrowing
                        indice_of_indice_indice_list_for_regrowing = (
                            (reconstruction_error > 0).int().to(torch.int64)
                        )

                        indice_indice_for_regrowing = torch.gather(
                            indice_indice_list_for_regrowing,
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                        )

                        regrowing_indice = torch.gather(
                            regrowing_indices_block,
                            1,
                            indice_indice_for_regrowing.to(torch.int64),
                        )

                        regrowing_metric = DSnoT_metric.gather(
                            1, regrowing_indice.to(torch.int64)
                        )

                        indice_indice_list_for_regrowing.scatter_(
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                            indice_indice_for_regrowing
                            + update_num_for_regrowing.gather(
                                1, indice_of_indice_indice_list_for_regrowing
                            ),
                        )

                        # pruning
                        indice_of_indice_indice_list_for_pruning = (
                            (reconstruction_error < 0).int().to(torch.int64)
                        )

                        indice_indice_for_pruning = torch.gather(
                            indice_indice_list_for_pruning,
                            1,
                            indice_of_indice_indice_list_for_pruning,
                        )

                        pruning_indice = torch.gather(
                            pruning_indices_block,
                            1,
                            indice_indice_for_pruning.to(torch.int64),
                        )

                        pruning_metric = DSnoT_metric.gather(
                            1, pruning_indice.to(torch.int64)
                        )

                        indice_indice_list_for_pruning.scatter_(
                            1,
                            indice_of_indice_indice_list_for_pruning, 
                            indice_indice_for_pruning
                            + update_num_for_pruning.gather(
                                1, indice_of_indice_indice_list_for_pruning
                            ),
                        )

                        # change mask
                        reconstruction_error_after = (reconstruction_error + pruning_metric - regrowing_metric)

                        if args.without_same_sign:
                            update_mask = update_mask & (
                                abs(reconstruction_error) > args.update_threshold
                            )
                        else:
                            update_mask = (
                                update_mask& (abs(reconstruction_error) > args.update_threshold)
                                & (initialize_error_sign == torch.sign(reconstruction_error_after))
                            )

                        weight_mask.scatter_(1, pruning_indice, update_mask)
                        weight_mask.scatter_(1, regrowing_indice, ~update_mask)

                        reconstruction_error += torch.where(
                            update_mask,
                            pruning_metric,
                            torch.zeros_like(pruning_metric),
                        )
                        reconstruction_error -= torch.where(
                            update_mask,
                            regrowing_metric,
                            torch.zeros_like(regrowing_metric),
                        )

            subset[name].weight.data[weight_mask] = 0

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def llama_sequential_owl_magnitude(args, model, dataloader, dev, logger):
    logger.info("Starting...")

    all_layer_ratio=[]
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    logger.info("Ready.")

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    quantizers = {}
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        layer_wmetric=[]
        for name in subset:
            logger.info(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_wmetric.append(W_metric)

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            out_ratio_layer=check_outlier_mean(layer_wmetric, out_ratio)
            print("layer outlier ratio", out_ratio, out_ratio_layer)
        
        all_layer_ratio.append(out_ratio_layer)

    logger.info("before adjustment", all_layer_ratio)

    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    all_layer_ratio=all_layer_ratio - np.mean(all_layer_ratio) + (1-args.sparsity_ratio)
    print(all_layer_ratio, np.mean(all_layer_ratio), np.max(all_layer_ratio), np.min(all_layer_ratio))
    
    logger.info("after adjustment", all_layer_ratio)

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    ############################ prune ############################
    logger.info('Pruning Starting ...')
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    hf_device_map = model.hf_device_map

    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        subset = find_layers(layer)

        for name in subset:
            layer_sparsity_ratio= 1-all_layer_ratio[i]

            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if args.prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % args.prune_m == 0:
                        tmp = W_metric[:, ii:(ii+args.prune_m)].float()
                        W_mask.scatter_(1, ii+torch.topk(tmp, args.prune_n, dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

    torch.cuda.empty_cache()


def llama_sequential_owl_sparsegpt(args, model, dataloader, dev, logger):
    logger.info("Starting...")

    all_layer_ratio=[]
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    logger.info("Ready.")

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    quantizers = {}
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        layer_wmetric=[]
        for name in subset:
            logger.info(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_wmetric.append(W_metric)

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            out_ratio_layer=check_outlier_mean(layer_wmetric, out_ratio)
            print("layer outlier ratio", out_ratio, out_ratio_layer)
        
        all_layer_ratio.append(out_ratio_layer)

    logger.info("before adjustment", all_layer_ratio)

    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    all_layer_ratio=all_layer_ratio - np.mean(all_layer_ratio) + (1-args.sparsity_ratio)
    print(all_layer_ratio, np.mean(all_layer_ratio), np.max(all_layer_ratio), np.min(all_layer_ratio))
    
    logger.info("after adjustment", all_layer_ratio)

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    ############################ prune ############################
    logger.info('Pruning Starting ...')
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    logger.info('Ready.')
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = find_layers(layer)

        layer_sparsity_ratio= 1-all_layer_ratio[i]
        if layer_sparsity_ratio<=0:
            layer_sparsity_ratio=0.01

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        
        for name in gpts:
            print(i, name)
            logger.info('Pruning ...')
            gpts[name].fasterprune(
                layer_sparsity_ratio, 
                prunen=args.prune_n, 
                prunem=args.prune_m, 
                percdamp=args.percdamp, 
                blocksize=args.blocksize
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def llama_sequential_owl_wanda(args, model, dataloader, dev, logger):
    logger.info("Starting...")

    all_layer_ratio=[]
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    logger.info("Ready.")

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        layer_wmetric=[]
        for name in subset:
            logger.info(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_wmetric.append(W_metric)

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            out_ratio_layer=check_outlier_mean(layer_wmetric, out_ratio)
            print("layer outlier ratio", out_ratio, out_ratio_layer)
        all_layer_ratio.append(out_ratio_layer)

    logger.info("before adjustment", all_layer_ratio)

    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    all_layer_ratio=all_layer_ratio - np.mean(all_layer_ratio) + (1-args.sparsity_ratio)
    print(all_layer_ratio, np.mean(all_layer_ratio), np.max(all_layer_ratio), np.min(all_layer_ratio))
    
    logger.info("after adjustment", all_layer_ratio)

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    ############################ prune ############################
    logger.info('Pruning Starting ...')
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    logger.info('Ready.')
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_sparsity_ratio= 1-all_layer_ratio[i]
           
            if layer_sparsity_ratio<=0:
                layer_sparsity_ratio=0.01

            W_mask = (torch.zeros_like(W_metric)==1)  # initialize a mask to be all False
            if args.prune_n != 0:
                # Structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % args.prune_m == 0:
                        tmp = W_metric[:, ii:(ii+args.prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, args.prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio) > 0.01) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha
                        
                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    logger.info(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:, :int(W_metric.shape[1] * layer_sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
            
            subset[name].weight.data[W_mask] = 0 # set the pruned weights to zero        

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        del wrapped_layers
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def llama_sequential_owl_wanda_structure(args, model, dataloader, dev, logger):
    logger.info("Starting...")

    all_layer_ratio=[]
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    logger.info("Ready.")

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    quantizers = {}
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        layer_wmetric=[]
        for name in subset:
            logger.info(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_wmetric.append(W_metric)

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            out_ratio_layer=check_outlier_mean(layer_wmetric, out_ratio)
            print("layer outlier ratio", out_ratio, out_ratio_layer)
        
        all_layer_ratio.append(out_ratio_layer)

    logger.info("before adjustment", all_layer_ratio)

    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda))
    all_layer_ratio=all_layer_ratio - np.mean(all_layer_ratio)
    all_layer_ratio=np.round(all_layer_ratio)
    all_layer_ratio=args.prune_n - all_layer_ratio
    print(all_layer_ratio, np.mean(all_layer_ratio), np.max(all_layer_ratio), np.min(all_layer_ratio))
    
    logger.info("after adjustment", all_layer_ratio)

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    ############################ prune ############################
    logger.info('Pruning Starting ...')
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    logger.info('Ready.')
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = find_layers(layer)

        args.prune_n = int(all_layer_ratio [i])
        print('Layer {} prune_n {} prune_m {}'.format(i, args.prune_n, args.prune_m))

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_sparsity_ratio= 1-all_layer_ratio[i]

            if layer_sparsity_ratio<=0:
                layer_sparsity_ratio=0.01

            W_mask = (torch.zeros_like(W_metric)==1)  # initialize a mask to be all False

            if args.prune_n != 0:
                # Structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % args.prune_m == 0:
                        tmp = W_metric[:, ii:(ii+args.prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, args.prune_n, dim=1, largest=False)[1], True)
            
            subset[name].weight.data[W_mask] = 0 # set the pruned weights to zero        

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        del wrapped_layers
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def llama_sequential_prunerzero(args, model, dataloader, dev, logger, engine=None):
    logger.info("Starting...")



    # Load gradients
    with open(args.gradient_path, 'rb') as file:
        gradients = torch.load(
            args.gradient_path, map_location=torch.device('cpu'))

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    logger.info("Ready.")

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = PrunerZeroGPT(subset[name])


        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(i, name)
            logger.info("Pruning ...")
            
            indexed_name = f'{name}_layer_{i}'
            W = torch.abs(subset[name].weight.data)
            # For inference, change engine forward without X
            X = wrapped_layers[name].scaler_row.reshape((1,-1))
            G = gradients[indexed_name]

            W_metric = engine.forward(
                W.to(dtype=torch.float32),
                G.to(device=W.device, dtype=torch.float32),
                X.to(dtype=torch.float32)
            )
            assert W_metric is not None

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if args.prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % args.prune_m == 0:
                        tmp = W_metric[:, ii:(ii+args.prune_m)].float()
                        W_mask.scatter_(1, ii+torch.topk(tmp, args.prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:, :int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del wrapped_layers
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def llama_sequential_gradient(args, model, dataloader, dev, logger):
    layers = model.model.layers
    with open(args.gradient_path, 'rb') as file:
        gradients = torch.load(args.gradient_path, map_location=torch.device('cpu')) 
    
    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    for i in range(len(layers)):
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        subset = find_layers(layer)

        for name in subset:
            indexed_name = f"{name}_layer_{i}"
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if not args.gradient_inv:
                W_metric = W_metric.to(dtype=torch.float32) * torch.abs(gradients[indexed_name].to(device=W_metric.device)).to(dtype=torch.float32)#+ small_value)
            else:
                small_value = torch.tensor(1e-8, dtype=gradients[indexed_name].dtype, device=gradients[indexed_name].device)
                gradient_inv = 1 / (torch.abs(gradients[indexed_name]) + small_value)
                W_metric = W_metric.to(dtype=torch.float32) * gradient_inv.to(device=W_metric.device).to(dtype=torch.float32)
            W_mask = (torch.zeros_like(W)==1)
            if args.prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % args.prune_m == 0:
                        tmp = W_metric[:,ii:(ii+args.prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, args.prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            W[W_mask] = 0


def llama_sequential_gblm_pruner(args, model, dataloader, dev, logger):
    with open(args.gradient_path, 'rb') as file:
        gradients = torch.load(args.gradient_path, map_location=torch.device('cpu')) 

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    logger.info("Ready.")

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    quantizers = {}
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = GBLMPrunerGPT(subset[name])
        
        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name))) ## this is a important function.
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove() 

        for name in subset:
            print(i, name)
            logger.info("Pruning ...")
            
            indexed_name = f'{name}_layer_{i}'
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            if not args.gradient_inv:
                # small_value = torch.tensor(1e-8, dtype=gradients[indexed_name].dtype, device=gradients[indexed_name].device)
                W_metric_grad = torch.abs(subset[name].weight.data)* torch.abs(gradients[indexed_name].to(device=W_metric.device))
                W_metric = W_metric.to(dtype=torch.float32) + W_metric_grad.to(dtype=torch.float32)  #+ small_value)
            else:
                small_value = torch.tensor(1e-8, dtype=gradients[indexed_name].dtype, device=gradients[indexed_name].device)
                gradient_inv = 1 / (torch.abs(gradients[indexed_name]) + small_value)
                W_metric = W_metric.to(dtype=torch.float32)  * gradient_inv.to(device=W_metric.device).to(dtype=torch.float32) 

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if args.prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % args.prune_m == 0:
                        tmp = W_metric[:, ii:(ii+args.prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, args.prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0
        
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def llama_sequential_flap(args, model, dataloader, dev, logger):
    logger.info("Starting...")

    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache['position_ids']

    logger.info('Ready.')

    attn_metric_list, mlp_metric_list = [], []
    attn_baseline_inp_list, mlp_baseline_inp_list = [], []
    attn_mask, mlp_mask = [], []

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        wrapped_layers = {}
        for name in subset:
                wrapped_layers[name] = FLAPGPT(subset[name], args.metrics)            

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            if name == 'self_attn.o_proj':
                W_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                if args.structure == "UL-UM":
                    W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                    thresh = torch.sort(W_metric.cuda())[0][int(args.sparsity_ratio*layer.self_attn.num_heads)].cpu()
                    W_mask = (W_metric>=thresh)
                    attn_mask.append(W_mask)
                elif args.structure == "UL-MM":
                    W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                    thresh = torch.sort(W_metric.cuda())[0][args.remove_heads // len(layers)].cpu()
                    W_mask = (W_metric>=thresh)
                    attn_mask.append(W_mask)
                else:
                    attn_metric_list.append(W_metric.cpu())
                attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            else:
                W_metric = metrics[args.metrics](wrapped_layers, subset, name)
                if args.structure == "UL-UM":
                    thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.sparsity_ratio)].cpu()
                    W_mask = (W_metric>=thresh)
                    mlp_mask.append(W_mask)
                elif args.structure == "UL-MM":
                    thresh = torch.sort(W_metric.cuda())[0][cal_remove_neuron(args, model)].cpu()
                    W_mask = (W_metric>=thresh)
                    mlp_mask.append(W_mask)
                else:
                    mlp_metric_list.append(W_metric.cpu())
                mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            wrapped_layers[name].free()

        inps, outs = outs, inps # Use the original output as input to the next layer
        torch.cuda.empty_cache()
    
    standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)

    if args.structure in ["AL-MM", "AL-AM"]:
        attn_metric = torch.stack(attn_metric_list)
        attn_metric = standarlization(attn_metric)
        attn_metric = attn_metric.reshape(len(layers), -1, 128).mean(dim=2)
        
        mlp_metric = torch.stack(mlp_metric_list)
        mlp_metric = standarlization(mlp_metric)
        
        if args.structure == "AL-MM":
            sorted_attn = torch.sort(attn_metric.view(-1), descending=True)[0]
            attn_thres = sorted_attn[-int(args.remove_heads)]
            attn_mask = (attn_metric > attn_thres)  # 1 means retain
            
            sorted_mlp = torch.sort(mlp_metric.view(-1), descending=True)[0]
            mlp_thres = sorted_mlp[-cal_remove_neuron(args, model)]
            mlp_mask = (mlp_metric > mlp_thres)
        else:
            prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
            sorted_prune, indices = torch.sort(prune_metric, descending=True)
            compression_weight = torch.ones_like(indices)
            compression_weight[indices < attn_metric.numel()] = 512.0 / 3
            threshold = sorted_prune[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*(1 - args.sparsity_ratio)))]
            attn_mask = (attn_metric > threshold)
            mlp_mask = (mlp_metric > threshold)
    else:
        attn_mask = torch.stack(attn_mask) 
        mlp_mask = torch.stack(mlp_mask)
    
    for idx in range(len(layers)):
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}): 
            compress(model.model.layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None, model.hf_device_map[f"model.layers.{idx}"], unstr=args.unstr)
        else:
            compress(model.model.layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None, DEV, unstr=args.unstr)
                
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}): 
            compress(model.model.layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx], model.hf_device_map[f"model.layers.{idx}"], unstr=args.unstr)
        else:
            compress(model.model.layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx], DEV, unstr=args.unstr)
                
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def llama_sequential_admm(args, model, dataloader, dev, logger):
    logger.info("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    logger.info("Ready.")

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    quantizers = {}
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = find_layers(layer)
        gpts = {}
        for name in subset:
            gpts[name] = AdmmGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(i, name)
            logger.info("Pruning ...")
            gpts[name].fasterprune(
                args.sparsity_ratio,
                prune_n=args.prune_n,
                prune_m=args.prune_m,
                percdamp=args.percdamp,
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def llama_sequential_RIA(args, model, dataloader, dev, logger):
    logger.info("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    logger.info("Ready.")

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    quantizers = {}
    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        subset = find_layers(layer)
        gpts = {}
        for name in subset:
            gpts[name] = RIAGPT(args, subset[name], layer_name=name, reconstruct=args.reconstruction)
            if args.gptq:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(i, name)
            if args.gptq:
                print('Quantizing ...')
                gpts[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )

            logger.info("Pruning ...")
            W = subset[name].weight.data.clone()
            W_metric = (torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * (torch.sqrt(gpts[name].scaler_row.reshape((1,-1))))**0.5
            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False

            if args.prune_n != 0:
                # structured n:m sparsity
                if args.reallocation:
                    """
                        Using Heuristic Channel Reallocation
                    """
                    
                    # Try with directly N:M sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % args.prune_m == 0:
                            tmp = W_metric[:,ii:(ii+args.prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, args.prune_n, dim=1, largest=False)[1], True)
                    
                    pre_score = torch.sum(W_metric[W_mask==0].type(torch.float32)).item()
                    print("The total value before resort: ", pre_score)
                    
                    # assign importance score to each columns
                    if args.importance_score == "sum":
                        # sum the total value of each column
                        sorted_idx = torch.sort(torch.sum(W_metric, dim=0))[1]
                    elif args.importance_score == "retained_degree_unstructured":
                        # try unstructured pruning
                        thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0]* W.shape[1]*args.sparsity_ratio)].cpu()
                        W_mask = (W_metric<=thresh)
                        keys = [torch.sum(W_mask, dim=0), torch.sum((W_mask==0)*W_metric, dim=0)]
                        sorted_idx = lexsort(keys)
                    elif args.importance_score == "retained_degree_per_outneuron":
                        # try unstructured pruning with per output neuron pruning
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)
                        indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                        W_mask = torch.zeros_like(W_metric)==1
                        W_mask.scatter_(1, indices, True)
                        
                        keys = [torch.sum(W_mask, dim=0), torch.sum((W_mask==0)*W_metric, dim=0)]
                        sorted_idx = lexsort(keys)
                    
                    # channel reallocation
                    index = torch.zeros_like(sorted_idx)
                    for ii in range(1, prune_m+1):
                        if ii % 2 == 1:
                            index[ii-1::args.prune_m] = sorted_idx[int(W_metric.shape[1]* (ii-1)/args.prune_m) :int(W_metric.shape[1]* ii/args.prune_m)]
                        else:
                            index[ii-1::args.prune_m] = sorted_idx[int(W_metric.shape[1]* (ii-1)/args.prune_m) :int(W_metric.shape[1]* ii/args.prune_m)].flip(0)
        
                    W_metric_resort = W_metric[:, index].clone()
                    
                    W_strip_value = torch.zeros(W_metric.shape[1]//args.prune_m).to(device)
                    W_mask_permute = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                    for ii in range(W_metric.shape[1]):
                        if ii % args.prune_m == 0:
                            tmp = W_metric_resort[:,ii:(ii+args.prune_m)].float()
                            W_mask_permute.scatter_(1,ii+torch.topk(tmp, args.prune_n, dim=1, largest=False)[1], True)
                            W_metric_strip = W_metric_resort[:, ii:(ii+args.prune_m)]
                            W_strip_value[ii//args.prune_m] = torch.sum(W_metric_strip[W_mask_permute[:, ii:(ii+args.prune_m)]==0])
                        
                    after_score = torch.sum(W_strip_value.type(torch.float32)).item()
                    print("The total value after heuristic channel reallocation: ", after_score)

                    if args.lsa:
                        """
                            Using linear sum assignment to finetune the N:M
                        """
                        permutation_device = "cuda:7"
                        if args.fast:
                            print("Use Fast!!")
                            fast_name_list = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
                            if name in fast_name_list:
                                blocks = 4
                            elif "up_proj" in name or "gate_proj" in name:
                                blocks = 8
                            else:
                                blocks = 16
                        else:
                            blocks = 1

                        shape = W_metric.shape[1]//args.prune_m//blocks
                        rows = torch.arange(shape).to(permutation_device)
                        lsa_columns = torch.arange(args.prune_m).to(permutation_device)
                        def lsa(W_metric, lsa_column, shape, rows, prune_n, prune_m, device):
                            W_metric = W_metric.to(device)
                            score_matrix = torch.zeros(shape, shape).to(device) # score matrix of LSA
                            num_parallel = 1 # How many parallel computation will be used.
                            
                            for row in range(shape//num_parallel):
                                strip_idx = torch.zeros(num_parallel, shape, prune_m).long().to(device)
                                block_columns = torch.arange(prune_m).to(device)
                                columns_mask = block_columns != lsa_column
                                block_columns = block_columns[columns_mask]
                                
                                strip_idx[:, :, 0] = (rows * prune_m).reshape(1, -1) + lsa_column
                                strip_idx[:, :, 1:] = block_columns.reshape(1, 1, -1) + torch.arange(row*num_parallel, (row+1)*num_parallel).reshape(-1, 1, 1).to(device) * prune_m
                                
                                tmp = W_metric[:, strip_idx].transpose(1, 0).transpose(2, 1)
                                
                                W_mask = torch.zeros_like(tmp).to(device)
                                
                                
                                
                                tmp_index = torch.sort(tmp, dim=-1)[1]
                                W_mask.scatter_(dim=-1, index=tmp_index[:, :, :, :prune_n], value=1)
                    
                                score_matrix[:, row*num_parallel:(row+1)*num_parallel] = torch.sum(torch.sum((tmp*(W_mask==0)), dim=-1), dim=-1).transpose(1, 0)
                            
                            score_matrix = score_matrix.transpose(1, 0)
                            
                            col_indices = torch.LongTensor(maximize_total_value(score_matrix.cpu())).to(device)
                            idx = torch.arange(W_metric.shape[1]).long().to(device)
                            idx[rows* prune_m + lsa_column] = col_indices * prune_m + lsa_column
                            
                            return idx
                        
                        z = 0
                        for lsa_column in lsa_columns:
                            t1 = time.time()
                            for ii in range(blocks):
                                index_tmp = index[ii*len(index)//blocks:(ii+1)*len(index)//blocks]
                                permute_idx = lsa(W_metric[:, index_tmp], lsa_column, shape, rows, args.prune_n, args.prune_m, permutation_device)
                                permute_idx = permute_idx.to(index.device)

                                index[ii*len(index)//blocks:(ii+1)*len(index)//blocks] = index_tmp[permute_idx]
                            t2 = time.time()
                            W_metric_permute = W_metric[:, index]
                            
                            W_mask_permute = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                            for ii in range(W_metric.shape[1]):
                                if ii % args.prune_m == 0:
                                    tmp = W_metric_permute[:,ii:(ii+args.prune_m)].float()
                                    W_mask_permute.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                                    W_metric_strip = W_metric_permute[:, ii:(ii+args.prune_m)]
                                    W_strip_value[ii//args.prune_m] = torch.sum(W_metric_strip[W_mask_permute[:, ii:(ii+args.prune_m)]==0])
                            print("The total value after linear sum assignment round {}: {}, running time: {}s".format(z, torch.sum(W_strip_value.type(torch.float32)).item(), round(t2-t1, 2)))
                            
                            z += 1

                    W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                    W_mask[:, index] = W_mask_permute
                    
                    if args.semi_sparse_acc and args.prune_n == 2 and args.prune_m == 4:
                        subset[name].weight = torch.nn.Parameter(to_sparse_semi_structured((W_mask_permute==0)*W[:, index].half()))
                        subset[name].mask = W_mask_permute==0
                    else:
                        subset[name].weight.data[W_mask] = 0

                else:
                    # Directly N:M
                    W_mask = (torch.zeros_like(W_metric) == 1)
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                    
                    subset[name].weight.data[W_mask] = 0
            else:
                if args.per_outneuron:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                else:
                    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0]* W.shape[1]*args.sparsity_ratio)].cpu()
                    W_mask = (W_metric<=thresh)
                    
                if args.reconstruction:
                    wrapped_layers[name].fasterprune(args.sparsity_ratio, mask=W_mask)
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero 
            gpts[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def llama_sequential_ALPS(args, model, dataloader, dev, logger):
    logger.info("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    logger.info("Ready.")

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        full = find_layers(layer)
        
        sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gpts = {}
            for name in subset:
                gpts[name] = ALPSGPT(subset[name], nsamples=args.nsamples, seqlen=model.seqlen)

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                logger.info("Pruning ...")
                gpts[name].ALPS_admm(
                    args.sparsity_ratio,
                    prunen=args.prune_n,
                    prunem=args.prune_m,
                    rho=args.rho
                )
                gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache


def llama_sequential_min_recon_error(args, model, dataloader, dev, logger):
    logger.info("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    prune_nsamples = args.nsamples

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = torch.float32
    inps = torch.zeros(
        (prune_nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    logger.info("Ready.")

    dense_inps = inps.clone()
    dense_outs = torch.zeros_like(inps)

    if args.initial_method == 'magnitude':
        from prunellm.min_recon_error.prune_fn import prune_magnitude
        prune_fn = prune_magnitude
    elif args.initial_method == 'wanda':
        from prunellm.min_recon_error.prune_fn import prune_wanda
        prune_fn = prune_wanda
    elif args.initial_method == 'sparsegpt':
        from prunellm.min_recon_error.prune_fn import prune_sparsegpt
        prune_fn = prune_sparsegpt
    
    dense_layers = []

    if args.use_cr:
        update_round = len(layers) + 1
        update_start = -1
    else:
        update_round = 1
        update_start = 0

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    for i in range(update_start, update_start + update_round):
        hf_device = f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)


        if args.use_cr:
            if i == -1:
                logger.info(f'pruning layer {i + 1}')
            elif i == (len(layers) - 1):
                logger.info(f'pruning layer {len(layers) - 1}')
            else:
                logger.info(f'pruning layer {i} & {i+1}')
        else:
            logger.info(f'pruning layer {i}')

        if args.use_cr:
            # For CR, save the current dense layer
            start_idx = max(0, i)
            end_idx = min(i + 1, len(layers) - 1)
            layer = layers[start_idx:end_idx + 1]
            if not (i ==  (len(layers) - 1)):
                dense_layer = type(layer[-1])(model.config).to(torch.float16).eval()
                dense_layer.load_state_dict(layer[-1].state_dict())
                dense_layers.append(dense_layer)
                dense_layer = dense_layer.float()
        else:
            layer = layers[i:i+1]
        layer = layer.float()

        subset = find_layers(layer)
        
        wrapped_layers = {}

        for name in subset:
            subset[name].prune_rate = 0
            if args.initial_method == 'magnitude':
                wrapped_layers[name] = None
            if args.initial_method == 'wanda':
                from prunellm.min_recon_error.wanda import WandaGPT
                wrapped_layers[name] = WandaGPT(subset[name], hf_device)
            elif args.initial_method == 'sparsegpt':
                from prunellm.min_recon_error.sparsegpt import SparseGPT
                wrapped_layers[name] = SparseGPT(subset[name], hf_device)
        

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        layer = layer.to(hf_device)
        handles = []

        if args.initial_method in ['wanda', 'sparsegpt']:
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
        with torch.no_grad():
            for j in range(prune_nsamples):
                outs[j] = obtain_output(layer, inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask, position_ids=position_ids).to('cpu')
        for h in handles:
            h.remove()

        # obtain outputs for x_dense
        if args.use_cr:
            with torch.no_grad():
                for j in range(start_idx, end_idx + 1):
                    dense_layers[j] = dense_layers[j].to(hf_device)
                for j in range(0, prune_nsamples, args.infer_batch_size):
                    dense_outs[j:j+args.infer_batch_size] = obtain_output(dense_layers[start_idx:end_idx + 1], dense_inps[j:j+args.infer_batch_size].to(hf_device), attention_mask=attention_mask, position_ids=position_ids).to('cpu')
                for j in range(start_idx, end_idx + 1):
                    dense_layers[j] = dense_layers[j].to('cpu')
        else:
            with torch.no_grad():
                for j in range(prune_nsamples):
                    dense_outs[j] = layer[0](dense_inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask, position_ids=position_ids)[0].to('cpu')
        


        for name in subset:
            print(i, name)
            subset[name].mask = nn.Parameter(torch.zeros(subset[name].weight.shape, device=hf_device))
            prune_fn(subset[name], wrapped_layers[name], args.sparsity_ratio, args.prune_n, args.prune_m)
            subset[name].prune_rate = args.sparsity_ratio

        # perform reconstruction
        if args.use_gp:
            train(layer, inps, dense_outs, dataloader, args, hf_device, attention_mask=attention_mask, position_ids=position_ids)
        else:
            train(layer, inps, outs, dataloader, args, hf_device, attention_mask=attention_mask, position_ids=position_ids)
        
        # calculate recon error
        if not args.use_cr:
            recon_error = val(layer, inps, dense_outs, args, hf_device, attention_mask, position_ids)
            logger.info(f"recon error {recon_error}")

        with torch.no_grad():
            for name in subset:
                mask_copy1 = subset[name].mask.clone()
                subset[name].weight.data = mask_copy1 * subset[name].weight.data
                subset[name].prune_rate = 0

        torch.cuda.empty_cache()

        # calculate outputs for x_sparse after pruning
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                for j in range(0, prune_nsamples, args.infer_batch_size):
                    outs[j:j+args.infer_batch_size] = layer[0](inps[j:j+args.infer_batch_size].to(hf_device), attention_mask=attention_mask, position_ids=position_ids, device=hf_device)[0].to("cpu")

        if args.use_cr:
            if i < 0:
                dense_outs = dense_inps.clone()
                outs = inps.clone()
            else:
                with torch.no_grad():
                    with torch.amp.autocast("cuda"):
                        dense_layers[i] = dense_layers[i].to(hf_device)
                        for j in range(0, prune_nsamples, args.infer_batch_size):
                            dense_outs[j:j+args.infer_batch_size] = obtain_output(dense_layers[i:i+1], dense_inps[j:j+args.infer_batch_size].to(hf_device), attention_mask=attention_mask, position_ids=position_ids).to('cpu')
                        dense_layers[i] = dense_layers[i].to('cpu')

                recon_error = val(layer[0:1], inps, dense_outs, args, hf_device, attention_mask, position_ids)

                dense_layers[i] = None
        
                logger.info(f"recon error {recon_error}")
                    
        inps, outs = outs, inps
        dense_inps, dense_outs = dense_outs, dense_inps
        layer = layer.to('cpu')
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    

def check_outlier_mean(mask,threshold):
    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()
    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def ww_sparsity(args, model, device=torch.device("cuda:0"), s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0):
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers
    
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())

    layer_num_in_block = int(len(prunables) / len(blocks))

    metrics = np.load(f"{args.ww_metric_cache}/{args.ww_metric}.npy")
    
    if args.mapping_type == 'block_wise':
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) for i in range(0, len(metrics), layer_num_in_block)]
        metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
    
    # print("metric values:", metrics)
            
    scores = torch.tensor(metrics)
    prunables = torch.tensor(prunables)

    # linear mapping
    max = torch.max(scores)
    min = torch.min(scores)
    
    layerwise_pruning_ratios = (((scores - min) / (max - min)) * (s2 - s1) + s1)
    scaler = torch.sum(prunables) * args.sparsity_ratio / (torch.sum(prunables * layerwise_pruning_ratios))  
    layerwise_pruning_ratios = layerwise_pruning_ratios * scaler
    layerwise_pruning_ratios = layerwise_pruning_ratios.cpu().numpy().tolist()
    
    print(layerwise_pruning_ratios)
    return layerwise_pruning_ratios


def check_sparsity(args, model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prune_method", type=str, choices=["magnitude", "sparsegpt", "wanda", "sparsellm", "DSnoT", "owl", "gradient", 
        "gblm-pruner", "pruner-zero", "flap", "admm", "RIA", "alphapruning", "ALPS", "min_recon_error"]
    )
    parser.add_argument(
        "--model", type=str, help="LlaMA model to load"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["wikitext2", "ptb", "c4"], help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--log_dir", default="./log", type=str, help="direction of logging file."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp", type=float, default=0.01, help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--sparsity_ratio", type=float, default=0, help="Target sparsity ratio"
    )
    parser.add_argument(
        "--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"]
    )
    parser.add_argument(
        "--blocksize", type=int, default=128, help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--gmp", action="store_true", help="Whether to run the GMP baseline."
    )
    parser.add_argument(
        "--wbits", type=int, default=16, help="Whether to quantize as well."
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="prunellm all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="prunellm all layers with id < this."
    )
    parser.add_argument(
        "--prune_only", type=str, default="", help="prunellm only layers that contain this text.",
    )
    parser.add_argument(
        "--invert", action="store_true", help="Invert subset."
    )
    parser.add_argument(
        "--save_model", type=str, default="", help="Path to saved model."
    )
    parser.add_argument(
        "--true_sequential", action="store_true", help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        '--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix"
    )
    parser.add_argument(
        "--eval_zero_shot", action="store_true"
    )

    # For DSnoT
    parser.add_argument("--initial_method", type=str, choices=["wanda", "sparsegpt", "magnitude"])
    parser.add_argument('--max_cycle_time', type=int, default=50, help='Max cycle time.')
    parser.add_argument('--without_DSnoT', action="store_true", help="without DSnoT")
    parser.add_argument('--update_threshold', type=float, default=0.1, help='update threshold.')
    parser.add_argument('--pow_of_var_regrowing', type=float, default=1, help='The power of variance.')
    parser.add_argument('--pow_of_var_pruning', type=float, default=1, help='The power of variance.')
    parser.add_argument("--skip_layer", type=str, default="mlp", choices=["no_skip", "mlp", "self_attn"])
    parser.add_argument("--skip_sub_layer", type=str, default="no_skip", choices=["no_skip", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "fc1", "fc2", "out_proj"])
    parser.add_argument('--without_same_sign', type=bool, default=True, help="without same sign")

    # For OWL
    parser.add_argument("--Lamda", default=0.08, type=float, help="Lamda")
    parser.add_argument('--Hyper_m', type=float, default=3)

    # For GBLM-Pruner
    parser.add_argument('--gradient_inv', action='store_true', help='Use inverse of gradient')

    # For Pruner-Zero
    parser.add_argument("--gradient_path", type=str, default=None, help="Path to save the gradient.")
    parser.add_argument("--json_tree", type=str, default="./prunellm/prunerzero/best_tree.json", help="Path to load the json tree.")

    # For FLAP
    parser.add_argument('--remove_heads', type=int, default=8, help='Remove num_heads')
    parser.add_argument("--metrics", type=str, default="WIFV", choices=["IFV", "WIFV", "WIFN", 'N/A'])
    parser.add_argument("--structure", type=str, default="AL-AM", choices=["UL-UM", "UL-MM", "AL-MM", "AL-AM", 'N/A'])
    parser.add_argument('--unstr', action="store_true")

    # For RIA
    parser.add_argument("--gptq", action="store_true", help="use gptq or not")
    parser.add_argument("--act_order", action="store_true", help="quantize activations or not")
    parser.add_argument("--lsa", action="store_true", help="Linear Sum Assignment")
    parser.add_argument("--reconstruction", action="store_true", help="remaining weight reconstruction based on sparsegpt")
    parser.add_argument("--reallocation", action="store_true", help="Heuristic Channel Reallocation")
    parser.add_argument("--importance_score", type=str, default="sum", help="assign importance score for columns")
    parser.add_argument("--per_outneuron", action="store_true", help="pruning per outneuron. Wanda's tactic.")

    # For AlphaPruning
    parser.add_argument("--ww_metric", default="alpha_peak", type=str, help="the WW-based metric to ues.")
    parser.add_argument("--ww_metric_cache", default="./cache/")
    parser.add_argument("--epsilon", default=0.3, type=float, help="for pruning ratio allocation.")
    parser.add_argument("--mapping_type", default="block_wise", type=str, help="mapping type for pruning ratios allocation.")
    
    # For ALPS
    parser.add_argument('--rho', type=float, default=300.0, help='initial rho')
    
    # For Minimize reconstruction Error
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate for training.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer.')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Beta1 for Adam optimizer.')
    parser.add_argument('--adam_beta2', type=float, default=0.95, help='Beta2 for Adam optimizer.')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='Epsilon for Adam optimizer.')
    parser.add_argument('--max_grad_norm', type=float, default=1000.0, help='Max gradient norm for clipping.')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps for learning rate scheduler.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training.')
    parser.add_argument('--lr_scheduler', type=str, default='linear', help='Learning rate scheduler type.')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps.')
    parser.add_argument('--infer_batch_size', type=int, default=1, help='Number of gradient accumulation steps.')
    parser.add_argument('--use_gp', action='store_true', help='Whether to use GP.')
    parser.add_argument('--use_cr', action='store_true', help='Whether to use CR.')


    args = parser.parse_args()

    if args.initial_method != None:
        args.log_dir = f"{args.log_dir}/{args.prune_method}-{args.initial_method}-{args.model.split('/')[-1]}"
    else:
        args.log_dir = f"{args.log_dir}/{args.prune_method}-{args.model.split('/')[-1]}"
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    logger = create_logger(log_dir)
    logger.info(args)

    # Setting seeds for reproducibility
    torch.random.manual_seed(args.seed)

    args.prune_n, args.prune_m = 0, 0
    if args.sparsity_type != "unstructured" and args.sparsity_ratio != 0:
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        args.prune_n, args.prune_m = map(int, args.sparsity_type.split(":"))
        args.save_model = f"{args.save_model}/{args.prune_n}_{args.prune_m}"
    else:
        if args.initial_method != None:
            args.save_model = f"{args.save_model}/{args.initial_method}/{args.sparsity_type}"
        else:
            args.save_model = f"{args.save_model}/{args.sparsity_type}"


    logger.info(f"loading llm model: {args.model}")
    model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.prune_method == "magnitude":
        from eval import llama_eval

        tick = time.time()
        llama_sequential_magnitude(args, model, dataloader, DEV, logger)
        logger.info(f"Total time: {time.time() - tick}")

        logger.info("Check sparisity ratio ...")
        sparsity_ratio = check_sparsity(args, model)

        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Dataset:", dataset)
            llama_eval(model, testloader, DEV, dataset, logger)

        if args.save_model:
            model.save_pretrained(args.save_model)

    elif args.prune_method == "sparsegpt":
        from prunellm.sparsegpt import SparseGPT
        from prunellm.quant import Quantizer
        from eval import llama_eval

        logger.info("Pruning SparseGPT...") 
        tick = time.time()
        llama_sequential_sparsegpt(args, model, dataloader, DEV, logger)
        logger.info(f"Total time: {time.time() - tick}")

        logger.info("Check sparisity ratio ...")
        sparsity_ratio = check_sparsity(args, model)
        
        logger.info("PPL Evaluation") 
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            logger.info(f"Dataset: {dataset}")
            llama_eval(args, model, testloader, DEV, dataset, logger)

        if args.save_model:
            model.save_pretrained(args.save_model)

    elif args.prune_method == "wanda":
        from prunellm.wanda import WandaGPT
        from eval import llama_eval

        logger.info("Pruning Wanda ...") 
        tick = time.time()
        llama_sequential_wanda(args, model, dataloader, DEV, logger)
        logger.info(f"Total time: {time.time() - tick}")

        logger.info("PPL Evaluation")    
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Dataset:", dataset)
            llama_eval(args, model, testloader, DEV, dataset, logger) 
        
        logger.info("Check sparisity ratio ...")
        sparsity_ratio = check_sparsity(args, model)
        
        # if args.eval_zero_shot:
        #     from eval import eval_zero_shot
        #     from transformers import AutoTokenizer
            
        #     task_list = ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
        #     num_shot = 0
        #     results = eval_zero_shot(args.model, model, logger, task_list, num_shot)
        #     logger.info("zero_shot evaluation")
        #     logger.info(results)

        if args.save_model:    
            model.save_pretrained(args.save_model)

    elif args.prune_method == "sparsellm":
        from prunellm.sparsellm import SparseLLM
        from eval import llama_eval

        logger.info("Pruning SparseLLM ...") 
        tick = time.time()
        llama_sequential_sparsellm(args, model, dataloader, DEV, logger)
        logger.info(f"Total time: {time.time() - tick}")

        logger.info("Check sparisity ratio ...")
        sparsity_ratio = check_sparsity(args, model)

        logger.info("PPL Evaluation")    
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Dataset:", dataset)
            llama_eval(args, model, testloader, DEV, dataset, logger) 
        
        if args.save_model:    
            model.save_pretrained(args.save_model)

    elif args.prune_method == "DSnoT":
        from prunellm.DSnoT import DSnoTGPT, return_reorder_indice
        from eval import llama_eval

        logger.info("Pruning DSnoT ...") 
        tick = time.time()
        llama_sequential_DSnoT(args, model, dataloader, DEV, logger)
        logger.info(f"Total time: {time.time() - tick}")

        logger.info("Check sparisity ratio ...")
        sparsity_ratio = check_sparsity(args, model)

        logger.info("PPL Evaluation")    
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            logger.info("Dataset:", dataset)
            llama_eval(args, model, testloader, DEV, dataset, logger) 
        
        if args.save_model:    
            model.save_pretrained(args.save_model)

    elif args.prune_method == "owl":
        from prunellm.owl import WrappedGPT
        from eval import llama_eval

        logger.info("Pruning Owl ...") 
        tick = time.time()
        if args.prune_method == "owl" and args.sparsity_type == "unstructured" and args.initial_method == 'magnitude':
            llama_sequential_owl_magnitude(args, model, dataloader, DEV, logger)
        if args.prune_method == "owl" and args.sparsity_type == "unstructured" and args.initial_method == 'sparsegpt':
            from prunellm.sparsegpt import SparseGPT
            llama_sequential_owl_sparsegpt(args, model, dataloader, DEV, logger)
        elif args.prune_method == "owl" and args.sparsity_type == "unstructured" and args.initial_method == 'wanda':
            llama_sequential_owl_wanda(args, model, dataloader, DEV, logger)
        elif args.prune_method == "owl" and args.sparsity_type != "unstructured" and args.initial_method == 'wanda':
            llama_sequential_owl_wanda_structure(args, model, dataloader, DEV, logger)
        logger.info(f"Total time: {time.time() - tick}")

        logger.info("Check sparisity ratio ...")
        sparsity_ratio = check_sparsity(args, model)

        logger.info("PPL Evaluation")    
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Dataset:", dataset)
            llama_eval(args, model, testloader, DEV, dataset, logger) 
        
        if args.save_model:    
            model.save_pretrained(args.save_model)

    elif args.prune_method == "gradient":
        from eval import llama_eval

        logger.info("Pruning Gradient ...") 
        tick = time.time()
        llama_sequential_gradient(args, model, dataloader, DEV, logger)
        logger.info(f"Total time: {time.time() - tick}")

        logger.info("Check sparisity ratio ...")
        sparsity_ratio = check_sparsity(args, model)

        logger.info("PPL Evaluation")    
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Dataset:", dataset)
            llama_eval(args, model, testloader, DEV, dataset, logger) 
        
        if args.save_model:    
            model.save_pretrained(args.save_model)

    elif args.prune_method == "gblm-pruner":
        from prunellm.gblm_pruner import GBLMPrunerGPT
        from eval import llama_eval

        logger.info("Pruning GBLM-Pruner ...") 
        tick = time.time()
        llama_sequential_gblm_pruner(args, model, dataloader, DEV, logger)
        logger.info(f"Total time: {time.time() - tick}")

        logger.info("Check sparisity ratio ...")
        sparsity_ratio = check_sparsity(args, model)

        logger.info("PPL Evaluation")    
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Dataset:", dataset)
            llama_eval(args, model, testloader, DEV, dataset, logger) 
        
        if args.save_model:    
            model.save_pretrained(args.save_model)

    elif args.prune_method == "pruner-zero":
        from prunellm.pruner_zero import PrunerZeroGPT
        from prunellm.prunerzero.gptree import GPTree
        from eval import llama_eval

        logger.info("Pruning PrunerZero ...") 
        tick = time.time()
        engine = GPTree.load_tree(args.json_tree)
        llama_sequential_prunerzero(args, model, dataloader, DEV, logger, engine)
        logger.info(f"Total time: {time.time() - tick}")

        logger.info("Check sparisity ratio ...")
        sparsity_ratio = check_sparsity(args, model)

        logger.info("PPL Evaluation")    
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Dataset:", dataset)
            llama_eval(args, model, testloader, DEV, dataset, logger) 
        
        if args.save_model:
            model.save_pretrained(args.save_model)

    elif args.prune_method == "flap":
        from prunellm.flap import FLAPGPT, cal_remove_neuron, compress, metrics
        from eval import llama_eval

        logger.info("Pruning Flap ...") 
        tick = time.time()
        llama_sequential_flap(args, model, dataloader, DEV, logger)
        logger.info(f"Total time: {time.time() - tick}")

        logger.info("Check sparisity ratio ...")
        sparsity_ratio = check_sparsity(args, model)

        logger.info("PPL Evaluation")    
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Dataset:", dataset)
            llama_eval(args, model, testloader, DEV, dataset, logger) 
        
        if args.save_model:    
            model.save_pretrained(args.save_model)

    elif args.prune_method == "admm":
        from prunellm.Admm import AdmmGPT
        from eval import llama_eval

        logger.info("Pruning ADMM ...") 
        tick = time.time()
        llama_sequential_admm(args, model, dataloader, DEV, logger)
        logger.info(f"Total time: {time.time() - tick}")

        logger.info("Check sparisity ratio ...")
        sparsity_ratio = check_sparsity(args, model)

        logger.info("PPL Evaluation")    
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Dataset:", dataset)
            llama_eval(args, model, testloader, DEV, dataset, logger) 
        
        if args.save_model:    
            model.save_pretrained(args.save_model)

    elif args.prune_method == "RIA":
        from prunellm.RIA import RIAGPT
        from eval import llama_eval

        logger.info("Pruning RIA ...") 
        tick = time.time()
        llama_sequential_RIA(args, model, dataloader, DEV, logger)
        logger.info(f"Total time: {time.time() - tick}")
        
        logger.info("Check sparisity ratio ...")
        sparsity_ratio = check_sparsity(args, model)

        logger.info("PPL Evaluation")    
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Dataset:", dataset)
            llama_eval(args, model, testloader, DEV, dataset, logger) 
        
        if args.save_model:    
            model.save_pretrained(args.save_model)

    elif args.prune_method == "alphapruning":
        from prunellm.esd_utils import get_esd_metrics
        from eval import llama_eval

        logger.info("Pruning AlphaPruning ...") 
        tick = time.time()

        if args.ww_metric_cache:
            args.ww_metric_cache = f"{args.ww_metric_cache}/{args.model.split('/')[-1]}"
            Path(args.ww_metric_cache).mkdir(parents=True, exist_ok=True)
            print("cache path: ", args.ww_metric_cache)
        # metric_values = get_esd_metrics(args, model, args.ww_metric)
        # np.save(f"{args.ww_metric_cache}/{args.ww_metric}.npy", metric_values)

        s1 = 1.0 - args.epsilon
        s2 = 1.0 + args.epsilon
        ratios = ww_sparsity(args, model, DEV, s1, s2)

        if args.initial_method == "wanda":
            from prunellm.wanda import WandaGPT
            llama_sequential_wanda(args, model, dataloader, DEV, logger, ratios=ratios)
        elif args.initial_method == "sparsegpt":
            from prunellm.sparsegpt import SparseGPT
            from prunellm.quant import Quantizer
            llama_sequential_sparsegpt(args, model, dataloader, DEV, logger, ratios=ratios)
        elif args.initial_method == "magnitude":
            llama_sequential_magnitude(args, model, dataloader, DEV, logger, ratios=ratios)

        logger.info(f"Total time: {time.time() - tick}")

        logger.info("Check sparisity ratio ...")
        sparsity_ratio = check_sparsity(args, model)

        logger.info("PPL Evaluation")    
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Dataset:", dataset)
            llama_eval(args, model, testloader, DEV, dataset, logger) 
        
        if args.save_model:    
            model.save_pretrained(args.save_model)

    elif args.prune_method == "ALPS":
        from prunellm.ALPS import ALPSGPT
        from eval import llama_eval

        logger.info("Pruning ALPS ...") 
        tick = time.time()
        llama_sequential_ALPS(args, model, dataloader, DEV, logger)
        logger.info(f"Total time: {time.time() - tick}")

        logger.info("Check sparisity ratio ...")
        sparsity_ratio = check_sparsity(args, model)

        logger.info("PPL Evaluation")    
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Dataset:", dataset)
            llama_eval(args, model, testloader, DEV, dataset, logger) 
        
        if args.save_model:    
            model.save_pretrained(args.save_model)

    elif args.prune_method == "min_recon_error":
        from prunellm.min_recon_error.prune_fn import obtain_output, find_layers
        from prunellm.min_recon_error.finetune import train, val
        from eval import llama_eval

        logger.info("Pruning Minimize Reconstruction Error ...")
        tick = time.time()
        llama_sequential_min_recon_error(args, model, dataloader, DEV, logger)
        logger.info(f"Total time: {time.time() - tick}")

        logger.info("Check sparisity ratio ...")
        sparsity_ratio = check_sparsity(args, model)

        logger.info("PPL Evaluation")    
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Dataset:", dataset)
            llama_eval(args, model, testloader, DEV, dataset, logger) 
        
        if args.save_model:    
            model.save_pretrained(args.save_model)
