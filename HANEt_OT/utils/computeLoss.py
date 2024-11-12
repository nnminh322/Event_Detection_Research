import torch
from configs import parse_arguments
args = parse_arguments()

def compute_CLLoss(Adj_mask, reprs, matsize): # compute InfoNCELoss
    logits_cl = torch.div(torch.matmul(reprs, reprs.T), args.cl_temp)
    if args.sub_max:
        logits_max_cl, _ = torch.max(logits_cl, dim=-1, keepdim=True)
        logits_cl = logits_cl - logits_max_cl
    exp_logits_cl = torch.exp(logits_cl)
    denom_cl = torch.sum(exp_logits_cl * (1 - torch.eye(matsize).to(args.device)), dim = -1) 
    log_prob_cl = -torch.mean((logits_cl - torch.log(denom_cl)) * Adj_mask, dim=-1)
    return torch.mean(log_prob_cl)

def compute_loss_TI(p_wi, true_trig):
    loss_TI = 0.0
    for i in range(len(true_trig)):
        loss_TI += -torch.dot(true_trig[i],torch.log(p_wi[i])) + torch.dot((1-true_trig[i]),torch.log(1-p_wi[i]))

    return loss_TI / len(true_trig)