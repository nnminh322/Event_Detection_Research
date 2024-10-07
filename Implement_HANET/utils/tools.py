import json, os
import torch
from configs import parse_arguments

args = parse_arguments()
device = torch.device(
    args.cuda if torch.cuda.is_available() and args.cuda != "cpu" else "cpu"
)


def compute_CLLoss(Adj_mask, reprs, mat_size):
    logits_cl = torch.div(torch.mul(reprs, reprs.T), args.cl_temp)
    if args.sub_max:
        logits_max_cl, _ = torch.max(logits_cl, dim=-1, keepdim=True)
        logits_cl = logits_cl - logits_max_cl
    # more stable when compute softmax, ex: instead softmax([10,15,20]) -> softmax([-10,-5,0])
    exp_logits_cl = torch.exp(logits_cl)
    mask = 1 - torch.eye(mat_size)  # only compute difference reprs
    denom_logits = torch.sum(exp_logits_cl * mask.to(device), dim=-1)
    log_prop_cl = -torch.mean((logits_cl - torch.log(denom_logits)) * Adj_mask, dim=-1)
    return torch.mean(log_prop_cl[log_prop_cl > 0])
    # just a little mathematical transform of the log of the 2 CL loss :))) put pen to paper


def collect_from_json(dataset, root, split):
    default = ["train", "dev", "test"]
    if split == "train":
        pth = os.path.join(
            root,
            dataset,
            "perm" + str(args.perm_id),
            f"{dataset}_{args.task_num}task_{args.class_num // args.task_num}way_{args.shot_num}shot.{split}.jsonl",
        )
    elif split in ["dev", "test"]:
        pth = os.path.join(root, dataset, f"{dataset}.{split}.jsonl")
    elif split == "stream":
        pth = os.path.join(
            root,
            dataset,
            f"stream_label_{args.task_num}task_{args.class_num // args.task_num}way.json",
        )
    else:
        raise ValueError(f'Split "{split}" value wrong!')
    if not os.path.exists(pth):
        raise FileNotFoundError(f"Path {pth} do not exist!")
    else:
        with open(pth) as f:
            if pth.endswith(".jsonl"):
                data = [json.loads(line) for line in f]
                if split == "train":
                    data = [list(i.values()) for i in data]
            else:
                data = json.load(f)
    return data
