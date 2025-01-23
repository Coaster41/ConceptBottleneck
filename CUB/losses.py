import torch
import numpy as np

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

def get_corr_to_std_torch(batch_size, std=1, mean=0, samples=100):
    min_corr = -1
    max_corr = 1
    step = 0.01
    scale_log = 2
    scale = 10**scale_log
    corr_to_std = []
    remapping_index = torch.arange(int(scale*min_corr), int((max_corr+step)*scale), int(step*scale)).cuda()

    for corr in range(int(scale*min_corr), int((max_corr+step)*scale), int(step*scale)):
        corr /= scale
        cov_matrix = np.ones((2,2))
        np.fill_diagonal(cov_matrix, std**2)
        cov_matrix[0,1] = corr*std**2
        cov_matrix[1,0] = corr*std**2
        z = np.random.multivariate_normal(np.ones(2)*mean, cov_matrix, (samples, batch_size))
        corr_z = []
        for sample in z:
            corr_z.append(np.corrcoef(sample, rowvar=False)[0,1])
        std_corr_z = np.std(corr_z)
        corr_to_std.append(std_corr_z)
        
    corr_to_std = torch.tensor(corr_to_std).cuda()

    def get_std(correlation): # interpolation
        index = torch.round(correlation*scale)
        bucket_index = torch.bucketize(index.ravel(), remapping_index)
        return corr_to_std[bucket_index].reshape(bucket_index.shape)
    return get_std

def get_cov_torch(c, get_std, delta=1):
    p_hat = torch.corrcoef(c.T) # 256+128, 256+128
    # P_hat = torch.cov(c.T)
    # eighvals_p_hat = torch.linalg.eigvals(p_hat)
    # eighvals_P_hat = torch.linalg.eigvals(P_hat)
    # # if torch.any(eighvals_p_hat < 0):
    # #     print('negative eigenvalues sample correlation')
    # # print(eighvals_p_hat)
    # # if torch.any(eighvals_P_hat < 0):
    # #     print('negative eigenvalues sample covariance')
    # # print(eighvals_P_hat)


    # calculate S_p
    S_p = torch.sum(get_std(p_hat)**2)**0.5


    # estimate gamma
    for gamma in range(2, 10, 2):
        p_hat_gamma = p_hat**gamma * p_hat
        # print('gamma:', gamma, torch.linalg.norm(p_hat - p_hat_gamma), delta * S_p)
        if torch.linalg.norm(p_hat - p_hat_gamma) >= delta * S_p:
            gamma_star = gamma
            break

    # estimate alpha
    scale = 1000
    step = 0.1
    for alpha in range(0, int((1+step)*scale), int(step*scale)):
        alpha /= scale
        L_alpha = alpha * p_hat ** gamma_star + (1-alpha) *p_hat ** (gamma_star-2)
        p_hat_alpha = L_alpha * p_hat
        # print('alpha:', alpha, torch.linalg.norm(p_hat - p_hat_alpha), delta * S_p)
        if torch.linalg.norm(p_hat - p_hat_alpha) >= delta * S_p:
            alpha_star = alpha - step
            break

    L_alpha_star = alpha_star * p_hat ** gamma_star + (1-alpha_star) *p_hat ** (gamma_star-2)
    p_hat_nice = L_alpha_star * p_hat

    V_hat = torch.diagflat(torch.std(c, axis=0))
    P_hat_nice = torch.matmul(torch.matmul(V_hat, p_hat_nice), V_hat) # Covariance estimate
    return P_hat_nice

def estimate_leakage_loss(c_hat, c_tilde, get_std, delta=1, m=100):
    c_hat = c_hat.T
    c_tilde = c_tilde.T
    c = torch.concatenate((c_hat, c_tilde), axis=1).squeeze() # 64, 256+128

    P_hat_nice = get_cov_torch(c, get_std)

    print(torch.sum(P_hat_nice), torch.logdet(P_hat_nice[-c_tilde.shape[1]:, -c_tilde.shape[1]:]),  torch.logdet(P_hat_nice))

    return -0.5 * (torch.logdet(P_hat_nice[-c_tilde.shape[1]:, -c_tilde.shape[1]:]) - torch.logdet(P_hat_nice))