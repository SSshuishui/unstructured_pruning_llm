import torch
import torch.nn as nn


def prune_preprocess_sparsegpt_aligned(W, X, device, lambda_reg=0.001):
    """
    与SparseGPT目标对齐的预处理
    核心思想：让Hessian矩阵更接近对角，便于剪枝
    """
    W_orig = W.clone()
    out_features, in_features = W.shape
    
    with torch.no_grad():
        # 计算Hessian近似 (X^T X)
        H = X.T @ X  # [in_features, in_features]
        
        # 计算重要性分数（与SparseGPT一致）
        X_norms = torch.norm(X, p=2, dim=0)
        diag_H = torch.diag(H)
        
        # SparseGPT使用的重要性度量
        importance = torch.abs(W) / torch.sqrt(diag_H.unsqueeze(0) + 1e-8)
        
        # 保守的权重调整：基于Hessian对角化的思想
        # 目标：让 W / sqrt(diag(H)) 的分布更易区分
        
        # 计算每行的统计量
        row_importance = importance.mean(dim=1)  # 每个输出通道的平均重要性
        
        # 非常保守的调整：仅对极端值做微小调整
        mean_imp = row_importance.mean()
        std_imp = row_importance.std()
        
        scaling_factors = torch.ones(out_features, device=device)
        for i in range(out_features):
            z_score = (row_importance[i] - mean_imp) / (std_imp + 1e-8)
            # 只对显著偏离的通道做微小调整
            if abs(z_score) > 2.0:  # 超过2个标准差
                adjustment = torch.tanh(z_score * 0.1) * 0.02  # 最大±2%
                scaling_factors[i] = 1.0 + adjustment
        
        print(f"Scaling factors - min: {scaling_factors.min():.4f}, max: {scaling_factors.max():.4f}, "
              f"mean: {scaling_factors.mean():.4f}")
        
        W_processed = W * scaling_factors.unsqueeze(1)
        
        # 验证输出变化
        XW_orig = X @ W_orig.T
        XW_processed = X @ W_processed.T
        rel_diff = torch.norm(XW_processed - XW_orig) / torch.norm(XW_orig)
        
        if rel_diff > 0.01:
            print(f"Output change too large: {rel_diff:.4f}, reverting")
            return W_orig
        
    return W_processed



def prune_preprocess_proximal(W, X, device, lambda_reg=0.005):
    """
    基于Proximal算子的理论方法
    """
    W_orig = W.clone()
    
    with torch.no_grad():
        n, m = W.shape  # n: output_features, m: input_features

        H = X.T @ X  # [m, m]
        
        # 理论目标函数：
        # min_{W'} 0.5 * ||XW - XW'||_F^2 + λ * φ(W')
        
        X_norms = torch.norm(X, p=2, dim=0)  # [m]
        diag_H = torch.diag(H)  # [m]
        
        # 构建正则化权重: 重要性低的权重惩罚更大
        reg_weights = X_norms / torch.sqrt(diag_H + 1e-8)  # [m]
        
        # Proximal gradient descent
        W_tilde = W.clone()
        L = torch.norm(X, 2) ** 2  # Lipschitz常数
        
        for _ in range(5):
            # 正确计算梯度: ∇f(W') = X^T(XW' - XW)
            # X: [batch_size, m], W: [n, m], W_tilde: [n, m]
            residual = X @ W_tilde.T - X @ W.T  # [batch_size, n]
            grad = X.T @ residual  # [m, n]
            grad = grad.T  # [n, m] - 转置回与W_tilde相同的形状
            
            # 梯度步
            W_temp = W_tilde - grad / L
            
            # Proximal操作：针对剪枝的软阈值
            thresholds = lambda_reg * reg_weights.unsqueeze(0) / L  # [1, m] -> [n, m]
            
            W_tilde = torch.sign(W_temp) * torch.clamp(
                torch.abs(W_temp) - thresholds, min=0
            )
        
        return W_tilde



def power_iteration(matrix, num_iterations=20):
    """使用幂迭代法快速估计矩阵的最大特征值"""
    vector = torch.randn(matrix.shape[1], 1, device=matrix.device)
    for _ in range(num_iterations):
        matrix_vector = matrix @ vector
        vector_norm = torch.norm(matrix_vector)
        vector = matrix_vector / vector_norm
    
    eigenvalue = (vector.T @ matrix @ vector) / (vector.T @ vector)
    return eigenvalue.item()

def prune_preprocess_proximal_optimized(
    W, 
    X, 
    device, 
    lambda_reg=0.001, 
    n_iter=100
):
    """
    优化后的基于Proximal算子的剪枝预处理方法
    """
    print("--- Starting OPTIMIZED PGD preprocessing for SparseGPT ---")
    W_orig = W.clone().to(device).float()
    X = X.to(device).float()
    n, m = W.shape

    with torch.no_grad():
        H = X.T @ X  # [m, m]

        # 1. 改进的正则化权重 (惩罚与重要性成反比)
        diag_H = torch.diag(H)
        # 数值稳定性处理
        safe_diag_H = torch.clamp(diag_H, min=1e-8)
        
        reg_weights = torch.sqrt(safe_diag_H)
        # 归一化，使lambda的尺度更可控
        reg_weights = reg_weights / torch.mean(reg_weights) if torch.mean(reg_weights) > 0 else reg_weights

        # 2. 改进的Lipschitz常数计算 (更准确且高效)
        print("Estimating Lipschitz constant (max eigenvalue of H)...")
        L = power_iteration(H)
        if L <= 0: L = 1.0 # 备用值
        step_size = 1.0 / L
        print(f"Lipschitz constant L ≈ {L:.4f}, using step_size ≈ {step_size:.6f}")

        # 3. PGD迭代 (增加迭代次数)
        W_tilde = W_orig.clone()
        
        # 预计算梯度不变的部分
        XtXW_orig_T = H @ W_orig.T # [m, n]
        
        for i in range(n_iter):
            # 正确计算梯度: ∇f(W') = (W' @ H) - (W_orig @ H)
            grad_T = H @ W_tilde.T - XtXW_orig_T
            grad = grad_T.T # 转置回 [n, m]
            
            # 梯度步
            W_temp = W_tilde - step_size * grad
            
            # Proximal操作
            thresholds = step_size * lambda_reg * reg_weights.unsqueeze(0)
            
            W_tilde = torch.sign(W_temp) * torch.clamp(
                torch.abs(W_temp) - thresholds, min=0
            )

        # 验证输出变化
        XW_orig_T = X @ W_orig.T
        XW_tilde_T = X @ W_tilde.T
        rel_diff = torch.norm(XW_tilde_T - XW_orig_T) / (torch.norm(XW_orig_T) + 1e-9)
        print(f"--- PGD finished. Relative Output Change: {rel_diff:.4f} ---")

        return W_tilde

