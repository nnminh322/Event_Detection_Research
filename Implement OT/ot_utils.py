import ot
import torch
import numpy as np


def compute_optimal_transport(p, q, C, epsilon=1e-8):
    batch_size, n, m = C.size()
    pi_star = []

    for i in range(batch_size):
        p_i = p[i].detach().cpu().numpy()
        q_i = q[i].detach().cpu().numpy()
        C_i = C[i].detach().cpu().numpy()

        pi_i = ot.sinkhorn(p_i, q_i, C_i, reg=epsilon)
        pi_star.append(pi_i)

    pi_star = np.stack(pi_star, axis=0)
    pi_star = torch.tensor(pi_star, dtype=torch.float, device=C.device)

    return pi_star
