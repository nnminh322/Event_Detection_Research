import torch
from configs import parse_arguments
import ot
import torch
import numpy as np

args = parse_arguments()


def compute_CLLoss(Adj_mask, reprs, matsize):  # compute InfoNCELoss
    logits_cl = torch.div(torch.matmul(reprs, reprs.T), args.cl_temp)
    if args.sub_max:
        logits_max_cl, _ = torch.max(logits_cl, dim=-1, keepdim=True)
        logits_cl = logits_cl - logits_max_cl
    exp_logits_cl = torch.exp(logits_cl)
    denom_cl = torch.sum(
        exp_logits_cl * (1 - torch.eye(matsize).to(args.device)), dim=-1
    )
    log_prob_cl = -torch.mean((logits_cl - torch.log(denom_cl)) * Adj_mask, dim=-1)
    return torch.mean(log_prob_cl)


def compute_loss_TI(p_wi, true_y):
    loss_TI = 0.0
    for i in range(len(true_y)):
        # mask padding token
        loss_TI += -torch.dot(true_y[i].float(), torch.log(p_wi[i])) - torch.dot(
            (1 - true_y[i].float()), torch.log(1 - p_wi[i])
        )

    return loss_TI / len(true_y)


def compute_loss_TP(p_tj, true_label):
    loss_TP = 0.0
    for i in range(len(true_label)):
        loss_TP += -torch.dot(true_label[i], torch.log(p_tj[i])) - torch.dot(
            (1 - true_label[i]), torch.log(1 - p_tj[i])
        )

    return loss_TP / len(true_label)


import torch
import torch.nn.functional as F

def compute_loss_Task(pi_star, y_true):
    """
    Compute binary cross-entropy loss for a batch with variable-length π* tensors.

    Args:
        pi_star (list of torch.Tensor): List of predicted alignment matrices,
                                        each of shape (num_words, num_labels).
        y_true (list of torch.Tensor): List of ground truth label indices,
                                       each of shape (num_words,).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    total_loss = 0.0
    total_words = 0  # Total number of words in the batch

    epsilon = 1e-12  # To avoid log(0)

    for pi_star_i, y_true_i in zip(pi_star, y_true):
        # Apply softmax to get probabilities (ensuring it's a valid probability distribution)
        pi_star_i = torch.nn.functional.softmax(pi_star_i, dim=1)
        print(f'pi_star_i.requires_grad: {pi_star_i.requires_grad}')
        # Create a binary vector for the true labels (one-hot encoded)
        y_true_one_hot = torch.zeros_like(pi_star_i)
        y_true_one_hot[torch.arange(len(y_true_i)), y_true_i] = 1

        # Compute the binary cross-entropy loss
        loss = F.binary_cross_entropy(pi_star_i, y_true_one_hot.float(), reduction='sum')

        total_loss += loss
        total_words += len(y_true_i)

    # Average over all words in the batch
    loss_Task = total_loss / total_words

    return loss_Task



def sinkhorn_pytorch(M, a, b, lambda_sh, numItermax=1000, stopThr=5e-3):
    u = torch.ones_like(a) / a.size(0)
    v = torch.zeros_like(b)
    K = torch.exp(-M * lambda_sh)

    cpt = 0
    err = 1.0

    def condition(cpt, u, v, err):
        return cpt < numItermax and err > stopThr

    def v_update(u, v):
        v = b / torch.matmul(K.t(), u)
        u = a / torch.matmul(K, v)
        return u, v

    def no_v_update(u, v):
        return u, v

    def err_f1(K, u, v, b):
        bb = v * torch.matmul(K.t(), u)
        err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float("inf"))
        return err

    def err_f2(err):
        return err

    def loop_func(cpt, u, v, err):
        u = a / torch.matmul(K, b / torch.matmul(u.T, K).T)
        cpt = cpt + 1
        if cpt % 20 == 1 or cpt == numItermax:
            u, v = v_update(u, v)
            err = err_f1(K, u, v, b)
        else:
            u, v = no_v_update(u, v)
            err = err_f2(err)
        return cpt, u, v, err

    while condition(cpt, u, v, err):
        cpt, u, v, err = loop_func(cpt, u, v, err)

    sinkhorn_divergences = torch.sum(u * torch.matmul(K * M, v), dim=0)
    return sinkhorn_divergences


def compute_loss_OT(Dist_pi_star, Dist_pi_g):
    return torch.abs(Dist_pi_star-Dist_pi_g).mean()
