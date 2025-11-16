import torch
import torch.nn as nn


def wanda_preprocess_gentle_adapted(W, scaler_row, beta=0.02):
    """
    适配Wanda框架的温和预处理
    """
    with torch.no_grad():
        n, m = W.shape
        X_norms = torch.sqrt(scaler_row)
        
        # 计算每行的平均重要性
        row_importance = (torch.abs(W) * X_norms.unsqueeze(0)).mean(dim=1)
        
        # 基于行重要性进行温和调整
        mean_imp = row_importance.mean()
        std_imp = row_importance.std()
        z_scores = (row_importance - mean_imp) / (std_imp + 1e-8)
        scaling_factors = 1.0 + beta * torch.tanh(z_scores)
        
        print(f"Gentle preprocessing - scaling range: [{scaling_factors.min():.4f}, {scaling_factors.max():.4f}]")
        
        W_processed = W * scaling_factors.unsqueeze(1)
        
        return W_processed



def get_weight_and_activation_norms(W, X):
    """
    Helper function to compute WANDA's importance components.
    
    Args:
        W (torch.Tensor): Weight matrix of shape (out_features, in_features).
        X (torch.Tensor): Calibration input activations of shape (num_samples, in_features).

    Returns:
        abs_W (torch.Tensor): Absolute weights, shape (out_features, in_features).
        act_norms (torch.Tensor): L2 norm of activations per input neuron, 
                                  shape (1, in_features) for broadcasting.
    """
    abs_W = torch.abs(W)
    # Compute L2 norm for each input neuron's activations
    act_norms = torch.norm(X.float(), p=2, dim=0, keepdim=True)
    return abs_W, act_norms


def solve_with_gradient_descent(W, X, grad_func, n_iter=50, lr=1e-3):
    """
    A generic solver for our optimization problems using gradient descent.
    
    Args:
        W (torch.Tensor): The original weight matrix.
        X (torch.Tensor): The input activation matrix.
        grad_func (function): A function that computes the gradient of the regularization term.
        n_iter (int): Number of iterations.
        lr (float): Learning rate.

    Returns:
        torch.Tensor: The optimized weight matrix W_prime.
    """
    W_prime = W.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([W_prime], lr=lr)
    
    # Precompute for fidelity term gradient
    XtX = X.T @ X
    XtXW = XtX @ W

    for _ in range(n_iter):
        optimizer.zero_grad()
        
        # Gradient of the fidelity term: 0.5 * ||XW' - XW||^2
        # Gradient is X^T(XW' - XW) = XtX @ W_prime - XtXW
        fidelity_grad = XtX @ W_prime - XtXW
        
        # Gradient of the regularization term, provided by grad_func
        reg_grad = grad_func(W_prime)
        
        # Total gradient
        total_grad = fidelity_grad + reg_grad
        
        W_prime.backward(total_grad)
        optimizer.step()
        
    return W_prime.detach()


def sceo_prune(W, X, sparsity_ratio, n_iter=50, lr=1e-4, lambda_reg=1.0, gamma=1.0):
    """
    Implements Sparsity-Aware Contrastive Enhancement Optimization (SCEO, Scheme 5).
    This function preprocesses weights to enhance the contrast between weights to be
    pruned and weights to be kept, based on a target sparsity ratio.

    Args:
        W (torch.Tensor): The original weight matrix (out_features, in_features).
        X (torch.Tensor): The calibration input activations (num_samples, in_features).
        sparsity_ratio (float): The target sparsity ratio for this layer (e.g., 0.5 for 50%).
        n_iter (int): Number of optimization iterations.
        lr (float): Learning rate for the optimization.
        lambda_reg (float): The overall strength of the contrastive regularization.
        gamma (float): The strength of the "reward" for weights to be kept.

    Returns:
        torch.Tensor: The preprocessed weight matrix W', ready for pruning.
    """
    print(f"Running SCEO Preprocessing for sparsity ratio: {sparsity_ratio:.2f}")

    if W.is_cuda:
        X = X.to(W.device)

    # Step 1: Identify survivor and victim sets based on original WANDA scores
    abs_W, act_norms = get_weight_and_activation_norms(W, X)
    importance_scores = abs_W * act_norms

    # Determine the threshold for the given sparsity ratio
    num_elements_to_prune = int(W.numel() * sparsity_ratio)
    if num_elements_to_prune == 0:
        print("Sparsity is 0, no preprocessing needed.")
        return W
    if num_elements_to_prune >= W.numel():
        print("Sparsity is 100%, returning zero matrix.")
        return torch.zeros_like(W)
        
    threshold = torch.kthvalue(importance_scores.view(-1), num_elements_to_prune).values

    # Create masks for the two sets
    mask_victims = (importance_scores <= threshold).float()
    mask_survivors = (importance_scores > threshold).float()
    
    # Define the gradient function for the contrastive regularization term
    def contrastive_gradient(W_prime):
        # Gradient of: lambda * (||Mask_V * W'||^2 - gamma * ||Mask_S * W'||^2)
        # Gradient wrt W' is: 2 * lambda * (Mask_V^2 * W' - gamma * Mask_S^2 * W')
        # Since masks are 0/1, Mask^2 = Mask.
        grad = 2 * lambda_reg * (mask_victims * W_prime - gamma * mask_survivors * W_prime)
        return grad

    # Step 2 & 3: Solve the optimization problem
    W_prime = solve_with_gradient_descent(W, X, contrastive_gradient, n_iter=n_iter, lr=lr)

    return W_prime



