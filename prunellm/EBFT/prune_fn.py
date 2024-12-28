import torch 
import torch.nn as nn 
import pandas as pd
from .finetune import train
from .linear_type import LinearMasked
from .wanda import WandaGPT
from .sparsegpt import SparseGPT
import time


def find_layers(module, layers=[nn.Linear], masked_layers=[LinearMasked], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
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


def llama_sequential_EBFT_magnitude(args, model, dataloader, dev, logger):
    logger.info("Starting...")

    table_dict = []
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
        subset = find_layers(layer)

        for name in subset:
            subset[name].prune_rate = 0
        layer.to(hf_device)

        with torch.no_grad():
            with train.amp.autocast("cuda"):
                for j in range(0, args.nsamples, args.infer_batch_size):
                    outs[j:j+args.infer_batch_size] = layer(inps[j:j+args.infer_batch_size].to(hf_device), attention_mask=attention_mask, position_ids=position_ids, device=hf_device)[0].to("cpu")
        for name in subset:
            W = subset[name].weight.data 
            M = subset[name].mask.data
            W_metric = torch.abs(W)
            if args.prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % args.prune_m == 0:
                        tmp = W_metric[:,ii:(ii+args.prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, args.prune_n, dim=1, largest=True)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*(args.sparsity_ratio))].cpu()
                W_mask = (W_metric>thresh)

            M[W_mask] = 1
            subset[name].prune_rate = args.sparsity_ratio
        init_loss, final_loss = train(layer, inps, outs, dataloader, args, hf_device, attention_mask=attention_mask, position_ids=position_ids, layer_index=i)
        table_dict.append({
            "init_loss": init_loss,
            "final_loss": final_loss,
            "flip_mask_num": 0,
            "layer_idx" : i,
        })
        with torch.no_grad():
            for name in subset:
                mask_copy1 = subset[name].mask.clone()
                subset[name].weight.data = mask_copy1 * subset[name].weight.data
                subset[name].prune_rate = 0
        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                for j in range(0, args.nsamples, args.infer_batch_size):
                    outs[j:j+args.infer_batch_size] = layer(inps[j:j+args.infer_batch_size].to(hf_device), attention_mask=attention_mask, position_ids=position_ids, device=hf_device)[0].to("cpu")
        inps, outs = outs, inps
        layer = layer.to('cpu')
        torch.cuda.empty_cache()
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    df = pd.DataFrame.from_records(table_dict, index="layer_idx")
    print(df)


def llama_sequential_EBFT_wanda(args, model, dataloader, dev, logger):
    logger.info("Starting...")

    table_dict = []
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

        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WandaGPT(subset[name], hf_device)
            subset[name].prune_rate = 0

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        layer.to(hf_device)

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                for j in range(0, args.nsamples, args.infer_batch_size):
                    outs[j:j+args.infer_batch_size] = layer(inps[j:j+args.infer_batch_size].to(hf_device), attention_mask=attention_mask, position_ids=position_ids, device=hf_device)[0].to("cpu")
        for h in handles:
            h.remove()

        wanda_mask_indices = {}
        for name in subset:
            print(i, name)
            logger.info("Pruning ...")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            if args.prune_n != 0:
                # structured n:m sparsity
                final_indices = None
                with torch.no_grad():
                    for ii in range(W_metric.shape[1]):
                        if ii % args.prune_m == 0:
                            tmp = W_metric[:,ii:(ii+args.prune_m)].float()
                            indices = ii+torch.topk(tmp, args.prune_n, dim=1, largest=True)[1].cuda()
                            subset[name].mask.scatter_(1, indices, 1)
                            if final_indices is None:
                                final_indices = indices.cpu()
                            else:
                                final_indices = torch.concat((final_indices,indices.cpu()),1)
                wanda_mask_indices[name] = final_indices
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True, descending=True)

                if args.use_variant:
                    # wanda variant 
                    print(f"no implementation")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*(1 - args.sparsity_ratio))]
                    wanda_mask_indices[name] = indices
                    with torch.no_grad():
                        subset[name].mask.scatter_(1, indices.cuda(), 1)

            subset[name].prune_rate = args.sparsity_ratio  ## set weights to zero 

        #通过重构误差来微调掩码
        init_loss, final_loss = train(layer, inps, outs, dataloader, args, hf_device, attention_mask=attention_mask, position_ids=position_ids, layer_index=i)
        table_dict.append({
            "init_loss": init_loss,
            "final_loss": final_loss,
            "flip_mask_num": 0,
            "layer_idx" : i,
        })
        with torch.no_grad():
            for name in subset:
                mask_copy1 = subset[name].mask.clone()
                subset[name].weight.data = mask_copy1 * subset[name].weight.data
                subset[name].prune_rate = 0
        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                for j in range(0, args.nsamples, args.infer_batch_size):
                    outs[j:j+args.infer_batch_size] = layer(inps[j:j+args.infer_batch_size].to(hf_device), attention_mask=attention_mask, position_ids=position_ids, device=hf_device)[0].to("cpu")
        inps, outs = outs, inps
        layer = layer.to('cpu')
        torch.cuda.empty_cache()
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    df = pd.DataFrame.from_records(table_dict, index="layer_idx")
    print(df)


def llama_sequential_EBFT_sparsegpt(args, model, dataloader, dev, logger):
    logger.info("Starting...")

    table_dict = []
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

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name], hf_device)
            subset[name].prune_rate = 0

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        layer.to(hf_device)

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=args.prune_n, prune_m=args.prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()
            subset[name].prune_rate = args.sparsity_ratio

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        #通过重构误差来微调掩码
        init_loss, final_loss = train(layer, inps, outs, dataloader, args, hf_device, attention_mask=attention_mask, position_ids=position_ids, layer_index=i)
        table_dict.append({
            "init_loss": init_loss,
            "final_loss": final_loss,
            "flip_mask_num": 0,
            "layer_idx" : i,
        })
        with torch.no_grad():
            for name in subset:
                mask_copy1 = subset[name].mask.clone()
                subset[name].weight.data = mask_copy1 * subset[name].weight.data
                subset[name].prune_rate = 0
        torch.cuda.empty_cache()
        starttime = time.time()
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0).to(hf_device), attention_mask=attention_mask, position_ids=position_ids)[0].to("cpu")
        endtime = time.time()
        print(endtime-starttime)
        layer = layer.to('cpu')
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    df = pd.DataFrame.from_records(table_dict, index="layer_idx")
    print(df)

