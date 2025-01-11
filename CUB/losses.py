import torch

def leakage_loss(c_hat, c_tilde):
    c_hat = c_hat.T.unsqueeze(-1)
    c_tilde = c_tilde.T.unsqueeze(-1)
    batch_size = c_hat.shape[0]
    c_hat_mean = torch.mean(c_hat, axis=0) # (n_concepts, 1)
    c_tilde_mean = torch.mean(c_tilde, axis=0) # (n_latent, 1)
    c_tilde_deviation = c_tilde-c_tilde_mean # (batch_size, n_latent, 1)
    c_tilde_deviation_T = torch.moveaxis(c_tilde_deviation,-1,-2) # (batch_size, 1, n_latent)
    sigma_tilde = 1/batch_size * torch.sum(torch.matmul(c_tilde_deviation, c_tilde_deviation_T),axis=0)  # (n_latent, n_latent)
    print("sigma_tilde", torch.linalg.det(sigma_tilde), torch.sum(sigma_tilde))
    log_det_sigma_tilde = torch.log(torch.linalg.det(sigma_tilde)) # scalar

    print("log_det_sigma_tilde", log_det_sigma_tilde)

    c_stack_T = torch.concatenate((c_hat,c_tilde), axis=1) # (batch_size, n_concepts+n_latent, 1)
    c_hat_mean_resize = torch.repeat_interleave(c_hat_mean.reshape(1,c_hat.shape[1],1), batch_size, axis=0) # (batch_size, n_concepts, 1)
    c_tilde_mean_resize = torch.repeat_interleave(c_tilde_mean.reshape(1,c_tilde.shape[1],1), batch_size, axis=0) # (batch_size, n_latent, 1)
    c_stack_mean_T = torch.concatenate((c_hat_mean_resize,c_tilde_mean_resize), axis=1)  # (batch_size, n_concepts+n_latent, 1)
    c_stack_deviation = c_stack_T-c_stack_mean_T  # (batch_size, n_concepts+n_latent, 1)
    c_stack_deviation_T = torch.moveaxis(c_stack_deviation,-1,-2)  # (batch_size, 1, n_concepts+n_latent)
    sigma = 1/batch_size * torch.sum(torch.matmul(c_stack_deviation,c_stack_deviation_T),axis=0)  # (batch_size, n_concepts+n_latent, n_concepts+n_latent)
    log_det_sigma = torch.log(torch.linalg.det(sigma)) # scalar

    print("log_det_sigma", log_det_sigma)

    return (log_det_sigma - log_det_sigma_tilde) / 2

def estimate_leakage_loss(c_hat, c_tilde, delta=1, m=100):
    c_hat = c_hat.T
    c_tilde = c_tilde.T
    batch_size = c_hat.shape[0]
    c = torch.concatenate((c_hat, c_tilde), axis=1).squeeze() # 64, 256+128
    concepts = c.shape[1]
    p_hat = torch.corrcoef(c.T) # 256+128, 256+128
    p_hat.fill_diagonal_(0.99999)
    
    mu = torch.arctanh(p_hat.flatten())
    sigma = torch.tensor([1/(batch_size-3)]).cuda() if torch.cuda.is_available() else torch.tensor(1/(batch_size-3))
    print(torch.repeat_interleave(mu, m).reshape(mu.shape[0],m))
    z_hat = torch.normal(torch.repeat_interleave(mu, m), torch.repeat_interleave(sigma,m*concepts**2)).reshape(m, concepts**2)
    p_var = torch.std(torch.tanh(z_hat), axis=0)
    S_p = torch.sqrt(torch.sum(p_var**2))

    # estimate gamma
    print(p_hat)
    print(batch_size, p_hat.shape, p_var.shape)
    print(p_var)
    print(p_var.mean(), p_var.min(), p_var.max(), p_var.std())
    for gamma in range(2, 100, 2):
        p_hat_gamma = p_hat**gamma * p_hat
        print(torch.norm(p_hat - p_hat_gamma), delta * S_p)
        if torch.norm(p_hat - p_hat_gamma) >= delta * S_p:
            gamma_star = gamma
            break
    print(torch.norm(p_hat))
    print(gamma_star)

    # estimate alpha
    scale = 1000
    step = 0.1
    for alpha in range(0, int((1+step)*scale), int(step*scale)):
        alpha /= scale
        L_alpha = alpha * p_hat ** gamma_star + (1-alpha) *p_hat ** (gamma_star-2)
        p_hat_alpha = L_alpha * p_hat
        if torch.norm(p_hat - p_hat_alpha) >= delta * S_p:
            alpha_star = alpha - step

    L_alpha_star = alpha_star * p_hat ** gamma_star + (1-alpha_star) *p_hat ** (gamma_star-2)
    p_hat_nice = L_alpha_star * p_hat

    V_hat = torch.diagflat(torch.std(c, axis=0))
    P_hat_nice = torch.matmul(torch.matmul(V_hat, p_hat_nice), V_hat) # Covariance estimate

    return -0.5 * torch.log(torch.linalg.det(P_hat_nice[-c_tilde.shape[1]:, -c_tilde.shape[1]:]) / torch.linalg.det(P_hat_nice))