import torch
from configs import parse_arguments
args = parse_arguments()

def compute_CLLoss(Adj_mask, reprs, matsize): # compute InfoNCELoss
    logits_cl = torch.div(torch.matmul(reprs, reprs.T), args.cl_temp)
    if args.sub_max:
        logits_max_cl, _ = torch.max(logits_cl, dim=-1, keepdim=True)
        logits_cl = logits_cl - logits_max_cl
    exp_logits_cl = torch.exp(logits_cl)
    denom_cl = torch.sum(exp_logits_cl * (1 - torch.eye(matsize).to(device)), dim = -1) 
    log_prob_cl = -torch.mean((logits_cl - torch.log(denom_cl)) * Adj_mask, dim=-1)
    return torch.mean(log_prob_cl)

def compute_loss_TI(p_wi, true_trig):
    loss_TI = 0.0
    loss_TI = -torch.sum(true_trig * torch.log(p_wi) + (1 - true_trig) * torch.log(1 - p_wi), dim=-1)
    
    # Trả về trung bình loss trên toàn bộ batch
    return loss_TI.mean()