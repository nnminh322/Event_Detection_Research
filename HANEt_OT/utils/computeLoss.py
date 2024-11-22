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


def compute_loss_Task(pi_star, y_true):
    """
    Compute L_task using negative log-likelihood over the predicted alignment matrix π*.

    Args:
        pi_star (torch.Tensor): Predicted alignment matrix (batch_size, n_labels).
        y_true (torch.Tensor): Ground truth labels for each word (batch_size,).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Lấy xác suất của nhãn đúng từ ma trận π*
    # torch.gather chọn giá trị tương ứng với y_true dọc theo chiều thứ 1
    batch_size = pi_star.size(0)
    true_probs = pi_star.gather(1, y_true.unsqueeze(1)).squeeze(1)  # (batch_size,)

    # Đảm bảo không có log(0) bằng cách thêm epsilon nhỏ
    epsilon = 1e-12
    true_probs = torch.clamp(true_probs, min=epsilon)

    # Tính negative log-likelihood loss
    loss_Task = -torch.log(true_probs).mean()  # Trung bình trên batch size

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
