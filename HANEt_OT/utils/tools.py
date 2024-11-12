import json, os
import torch
from configs import parse_arguments
import itertools

args = parse_arguments()
device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")  # type: ignore


def compute_CLLoss(Adj_mask, reprs, matsize):  # compute InfoNCELoss
    logits_cl = torch.div(torch.matmul(reprs, reprs.T), args.cl_temp)
    if args.sub_max:
        logits_max_cl, _ = torch.max(logits_cl, dim=-1, keepdim=True)
        logits_cl = logits_cl - logits_max_cl
    exp_logits_cl = torch.exp(logits_cl)
    denom_cl = torch.sum(exp_logits_cl * (1 - torch.eye(matsize).to(device)), dim=-1)
    log_prob_cl = -torch.mean((logits_cl - torch.log(denom_cl)) * Adj_mask, dim=-1)
    return torch.mean(log_prob_cl[log_prob_cl > 0])


def extract_single_dict(lst):
    for item in lst:
        if isinstance(item, list):
            return extract_single_dict(item)  # Gọi đệ quy cho các danh sách con
        elif isinstance(item, dict):
            return item  # Trả về từ điển đầu tiên tìm thấy


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
    # if split == "train":
    #     data = extract_single_dict(data)

    return data



def get_one_hot_true_label_and_true_trigger(data_instance, num_label):
    true_label = []
    true_trigger = []
    seq_len = len(data_instance["piece_ids"]) # because start_index of piece_ids is 1 instead of 0
    
    for i in range(len(data_instance["label"])):
        if data_instance["label"][i] != 0:
            true_label.append(data_instance["label"][i])
            true_trigger.append(data_instance["span"][i])


    true_one_hot_label_vector = torch.zeros(num_label)
    true_one_hot_trigger_vector = torch.zeros(seq_len)

    set_label_in_one_sentence = set([label.item() for label in true_label])
    for i in set_label_in_one_sentence:
        true_one_hot_label_vector += torch.eye(num_label)[i]


    list_trigger = [trigger.tolist() for trigger in true_trigger]
    trigger = []
    for i in list_trigger:
        trigger.extend(i)

    set_trig_in_one_sentence = set(trigger)

    for i in set_trig_in_one_sentence:
        true_one_hot_trigger_vector += torch.eye(seq_len)[i]
    true_one_hot_trigger_vector = true_one_hot_trigger_vector.to(device)
    true_one_hot_label_vector = true_one_hot_label_vector.to(device)
    return true_one_hot_trigger_vector, true_one_hot_label_vector


def true_label_and_trigger(train_x,train_y,train_masks, train_span, class_num):
    num_instance = len(train_x)
    true_one_hot_label_vectors = []
    true_one_hot_trigger_vectors = []
    for i in range(num_instance):
        data_instace={
            'piece_ids': train_x[i],
            'label': train_y[i],
            'span': train_span[i],
            'mask': train_masks[i]
        }

        true_one_hot_trigger_vector, true_one_hot_label_vector = get_one_hot_true_label_and_true_trigger(data_instance=data_instace,num_label=class_num)
        true_one_hot_trigger_vectors.append(true_one_hot_trigger_vector)
        true_one_hot_label_vectors.append(true_one_hot_label_vector)
    true_one_hot_trigger_vectors = [x.to(device) for x in true_one_hot_trigger_vectors]
    true_one_hot_label_vectors = [x.to(device) for x in true_one_hot_label_vectors]
    return true_one_hot_trigger_vectors, true_one_hot_label_vectors

        
