import torch

def W_gradient_preprocess_pruning(W, X, device, lambda_reg=0.01, lr=0.001, n_iter=200):
    """
    Simplified gradient-based preprocessing for pruning
    """
    W_hat = W.clone().T
    
    m, n = X.shape

    U, s, Vt = torch.linalg.svd(X)
    del X
    s /= torch.max(s)
    S = torch.diag(s)
    if m > n:
        U = U[:, :n]
    elif m < n:
        Vt = Vt[:m, :]

    X = torch.mm(torch.mm(U, S), Vt)
    XtX = torch.matmul(X.T, X).to(device)

    # Compute feature norms and precompute matrices
    with torch.no_grad():
        feature_norms = torch.norm(X, dim=0)  # (d, )
        feature_norms = feature_norms.to(device)
        D_sq = feature_norms ** 2  # (d, )
        
    
    for i in range(n_iter):
        # Manual gradient computation
        # Gradient of data term: 2 * X^T X (W_hat - W)
        grad_data = 2 * torch.matmul(XtX, (W_hat - W.T))
        
        # Gradient of regularization term: 2 * lambda_reg * D^2 W_hat
        grad_reg = 2 * lambda_reg * (D_sq.unsqueeze(1) * W_hat)
        
        # Total gradient
        grad_total = grad_data + grad_reg
        
        # Gradient descent step
        W_hat = W_hat - lr * grad_total
    
    return W_hat.T


def W_gradient_preprocess_pruning_per_group(W, X, device, groupsize=128, lambda_reg=0.01, lr=0.001, n_iter=200):
    """
    Group-wise preprocessing for pruning - applies regularization per output channel group
    
    Parameters:
    W: (d, m) weight matrix
    X: (n, d) input data
    groupsize: size of each group along output dimension
    lambda_reg: regularization strength
    lr: learning rate
    n_iter: number of iterations
    """
    W_hat = W.clone().T  # (m, d)
    
    m, n = X.shape

    # SVD compression (same as your original)
    U, s, Vt = torch.linalg.svd(X)
    del X
    s /= torch.max(s)
    S = torch.diag(s)
    if m > n:
        U = U[:, :n]
    elif m < n:
        Vt = Vt[:m, :]

    X = torch.mm(torch.mm(U, S), Vt)
    XtX = torch.matmul(X.T, X).to(device)

    # Compute feature norms
    with torch.no_grad():
        feature_norms = torch.norm(X, dim=0)  # (d, )
        feature_norms = feature_norms.to(device)
        D_sq = feature_norms ** 2  # (d, )
    
    # Precompute group information
    n_output, n_input = W_hat.shape  # (m, d)
    n_groups = (n_output + groupsize - 1) // groupsize
    
    for i in range(n_iter):
        # Gradient of data term: 2 * X^T X (W_hat - W.T)
        grad_data = 2 * torch.matmul(XtX, (W_hat - W.T))
        
        # Group-wise regularization gradient
        grad_reg = torch.zeros_like(W_hat)
        
        for group_idx in range(n_groups):
            start_idx = group_idx * groupsize
            end_idx = min((group_idx + 1) * groupsize, n_output)
            
            # Extract group weights: (group_size, d)
            group_weights = W_hat[start_idx:end_idx, :]
            
            if group_weights.numel() == 0:
                continue
                
            # Compute group importance scores (per output channel)
            # This is where we mimic MagR's group-wise regularization
            with torch.no_grad():
                # Method 1: Use feature norms to weight input dimensions
                group_importance = torch.norm(group_weights * feature_norms.unsqueeze(0), dim=1)  # (group_size,)
                group_importance = torch.clamp(group_importance, min=1e-8)
                
                # Normalize importance within group
                importance_weights = group_importance / torch.max(group_importance)
                
                # Invert for regularization: less important channels get more regularization
                reg_strength = (1.0 - importance_weights).unsqueeze(1)  # (group_size, 1)
            
            # Apply group-aware regularization
            # Channels with lower importance get stronger regularization
            group_grad_reg = 2 * lambda_reg * reg_strength * (D_sq.unsqueeze(0) * group_weights)
            grad_reg[start_idx:end_idx, :] = group_grad_reg
        
        # Total gradient
        grad_total = grad_data + grad_reg
        
        # Gradient descent step
        W_hat = W_hat - lr * grad_total
    
    return W_hat.T