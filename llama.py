import time
import torch
import torch.nn as nn
import argparse
from pathlib import Path

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


def llama_sequential_magnitude(args, model, dataloader, dev, logger):
    logger.info("Starting...")

    layers = model.model.layers 

    for i in range(len(layers)):
        logger.info(f'================={i}==================')
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
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0


@torch.no_grad()
def llama_sequential_sparsegpt(args, model, dataloader, dev, logger):
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
                sparsity = args.sparsity_ratio
                gpts[name].fasterprune(
                    sparsity,
                    prunen=args.prune_n,
                    prunem=args.prune_m,
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
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


def llama_sequential_wanda(args, model, dataloader, dev, logger):
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

        subset = find_layers(layer)

        wanda_layers = {}
        for name in subset:
            wanda_layers[name] = WandaGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wanda_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wanda_layers:
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
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
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
                    gpts[name].add_batch(inp[0].data, out.data)

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
                print(i, name)
                logger.info("Pruning ...")
                sparsity = args.sparsity_ratio
                gpts[name].fasterprune(
                    sparsity,
                    prunen=args.prune_n,
                    prunem=args.prune_m,
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                )
                gpts[name].free()

            # Adjust hyperparameters as needed
            alpha = 5.0
            beta = 5.0
            gamma = 5.0

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
            # z: output of 'mlp.up_proj'
            hidden_z_list = gpts['mlp.up_proj'].batch_out
            z = torch.stack(hidden_z_list, dim=0)
            hidden_z_list = None
            gpts['mlp.up_proj'].batch_out.clear()
            # p: input of 'mlp.down_proj'
            hidden_p_list = gpts['mlp.down_proj'].batch_inp
            p = torch.stack(hidden_p_list, dim=0)
            hidden_p_list = None
            gpts['mlp.down_proj'].batch_inp.clear()
            # s: output of 'mlp.gate_proj'
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
                    print('Pruning ...')
                    sparsity = args.sparsity
                    gpts[name].fasterprune(
                        sparsity,
                        prunen=args.prunen,
                        prunem=args.prunem,
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
                    )
                
                ##############
                # optimize p
                ##############
                # Activation inverse
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
                w = subset['mlp.up_proj'].weight
                m = torch.matmul(w, X)
                swish = nn.functional.silu(s)
                z = (m + swish * p) / (swish ** 2 + 1)    

                del w, m, swish
                torch.cuda.empty_cache()
                
                ##############
                # optimize s
                ##############
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prune_method", type=str, choices=["magnitude", "sparsegpt", "wanda", "sparsellm"]
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
        "--log_dir", default="./log/", type=str, help="direction of logging file."
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
        "--true-sequential", action="store_true", help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        '--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix"
    )
    parser.add_argument(
        "--eval_zero_shot", action="store_true"
    )

    args = parser.parse_args()

    args.log_dir = f"{args.log_dir}/{args.prune_method}-{args.model.split('/')[-1]}"
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    logger = create_logger(log_dir)
    logger.info(args)

    # Setting seeds for reproducibility
    torch.random.manual_seed(args.seed)

    args.prune_n, args.prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        args.prune_n, args.prune_m = map(int, args.sparsity_type.split(":"))


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

        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            logger.info("Dataset:", dataset)
            llama_eval(model, testloader, DEV, dataset, logger)

        if args.save_model:
            model.save_pretrained(args.save_model)

    if args.prune_method == "sparsegpt":
        from prunellm.sparsegpt import SparseGPT
        from prunellm.quant import Quantizer
        from eval import llama_eval

        logger.info("Pruning SparseGPT...") 
        tick = time.time()
        llama_sequential_sparsegpt(args, model, dataloader, DEV, logger)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'down_proj' in n:
                break
        logger.info(f"Total time: {time.time() - tick}")
        
        logger.info("PPL Evaluation") 
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            logger.info(f"Dataset: {dataset}")
            llama_eval(args, model, testloader, DEV, dataset, logger)

        if args.save_model:
            model.save_pretrained(args.save_model)

    if args.prune_method == "wanda":
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
            logger.info("Dataset:", dataset)
            llama_eval(args, model, testloader, DEV, dataset, logger) 
        
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

    if args.prune_method == "sparsellm":
        from prunellm.sparsellm import SparseLLM
        from eval import llama_eval

        logger.info("Pruning SparseLLM ...") 
        tick = time.time()
        llama_sequential_sparsellm(args, model, dataloader, DEV, logger)
        logger.info(f"Total time: {time.time() - tick}")

        logger.info("PPL Evaluation")    
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            logger.info("Dataset:", dataset)
            llama_eval(args, model, testloader, DEV, dataset, logger) 
        
        if args.save_model:    
            model.save_pretrained(args.save_model)