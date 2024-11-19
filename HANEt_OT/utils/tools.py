import json, os
import torch
from configs import parse_arguments
import itertools
import ot
import numpy as np

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
    seq_len = len(
        data_instance["piece_ids"]
    )  # because start_index of piece_ids is 1 instead of 0
    matrix_word_is_label = torch.zeros(seq_len, num_label, dtype=int)
    for i in range(len(data_instance["label"])):
        if data_instance["label"][i] != 0:
            true_label.append(data_instance["label"][i])
            true_trigger.append(data_instance["span"][i])
            for word_is_trigger in data_instance["span"][i]:
                matrix_word_is_label[word_is_trigger, data_instance["label"][i]] = 1

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
    return true_one_hot_trigger_vector, true_one_hot_label_vector, matrix_word_is_label


def true_label_and_trigger(train_x, train_y, train_masks, train_span, class_num):
    num_instance = len(train_x)
    true_one_hot_label_vectors = []
    true_one_hot_trigger_vectors = []
    golden_matrix = []
    for i in range(num_instance):
        data_instace = {
            "piece_ids": train_x[i],
            "label": train_y[i],
            "span": train_span[i],
            "mask": train_masks[i],
        }

        true_one_hot_trigger_vector, true_one_hot_label_vector, matrix_word_is_label = (
            get_one_hot_true_label_and_true_trigger(
                data_instance=data_instace, num_label=class_num
            )
        )
        true_one_hot_trigger_vectors.append(true_one_hot_trigger_vector)
        true_one_hot_label_vectors.append(true_one_hot_label_vector)
        golden_matrix.append(matrix_word_is_label)
    true_one_hot_trigger_vectors = torch.stack(
        [x.to(device) for x in true_one_hot_trigger_vectors]
    )
    true_one_hot_label_vectors = torch.stack(
        [x.to(device) for x in true_one_hot_label_vectors]
    )
    pi_golden_matrix = torch.stack([x.to(device) for x in golden_matrix])
    return true_one_hot_trigger_vectors, true_one_hot_label_vectors, pi_golden_matrix


def compute_single_optimal_transport_for_1_sentence(p, q, C, epsilon=0.05):
    # Đảm bảo các tensor p, q, C đều ở trên cùng một device (CPU hoặc GPU)
    device = p.device

    # Chuyển tensor PyTorch thành numpy nếu cần
    p_i = p.detach().cpu().numpy()
    q_i = q.detach().cpu().numpy()
    C_i = C.detach().cpu().numpy()

    # Sử dụng phương thức Sinkhorn từ thư viện optimal transport
    pi_i = ot.sinkhorn(p_i, q_i, C_i, reg=epsilon)

    # Chuyển lại kết quả thành tensor PyTorch và đưa về đúng device
    pi_i_tensor = torch.tensor(pi_i, device=device)

    return pi_i_tensor

def get_true_y(y, num_classes=args.class_num + 1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    true_trig, true_label = [], []

    for i in range(len(y)):
        true_label_loop = torch.zeros(num_classes, device=device)  # Đưa tensor này lên GPU
        set_label = set(y[i].tolist())
        
        for label in set_label:
            if label != 0:
                true_label_loop += torch.nn.functional.one_hot(
                    torch.tensor(label, device=device), num_classes=num_classes  # Đưa label lên GPU
                )

        filter_y = (y[i] != 0).int().to(device)  # Đưa filter_y lên GPU
        true_trig.append(filter_y)
        true_label.append(true_label_loop)
    
    return true_trig, true_label


def compute_cost_transport(last_hidden_state_order, label_embeddings, num_classes = args.class_num+1):
    # last_hidden_state_order: [batch_size, num_span, hidden_dim]
    # label_embedding: [num_class, hidden_dim]
    
    batch_size = len(last_hidden_state_order)
    cost_matrix = []
    for i in range(batch_size):
        num_span = last_hidden_state_order[i].size(0)
        label_embeddings_scale = label_embeddings.unsqueeze(0).repeat([num_span,1,1])
        last_hidden_state_order_scale = last_hidden_state_order[i].unsqueeze(1).repeat([1,num_classes,1])
        cost = 1-torch.nn.functional.cosine_similarity(last_hidden_state_order_scale,label_embeddings_scale,dim=-1)
        cost_matrix.append(cost)
    return cost_matrix

def compute_optimal_transport_plane_for_batch(D_W_P_order,D_T_P, cost_matrix):
    batch_size = len(D_W_P_order)
    pi_star_matrix = []
    for sentence in range(batch_size):
        pi_i = compute_single_optimal_transport_for_1_sentence(D_W_P_order[sentence],D_T_P[sentence],cost_matrix[sentence])
        pi_star_matrix.append(pi_i)
    
    return pi_star_matrix

