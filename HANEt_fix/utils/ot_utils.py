import ot
import torch
import numpy as np


def compute_optimal_transport(p, q, C, epsilon=0.1):
    batch_size, n, m = C.size()
    pi_star = []

    for i in range(batch_size):
        p_i = p[i].cpu().numpy()
        q_i = q[i].cpu().numpy()
        C_i = C[i].cpu().numpy()

        pi_i = ot.sinkhorn(p_i, q_i, C_i, reg=epsilon)
        pi_star.append(pi_i)

    pi_star = np.stack(pi_star,axis=0)
    pi_star = torch.tensor(pi_star, dtype=torch.float,device=C.device)

    return pi_star