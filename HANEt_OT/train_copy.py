import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
from torch.optim import AdamW
from utils import *
from configs import parse_arguments
from model import BertED
from tqdm import tqdm
from exemplars import Exemplars
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import os, time
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.computeLoss import *
from utils.tools import *


def train(local_rank, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s "
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    cur_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    if args.log:
        if not os.path.exists(
            os.path.join(
                args.tb_dir,
                args.dataset,
                args.joint_da_loss,
                str(args.class_num) + "class",
                str(args.shot_num) + "shot",
                args.cl_aug,
                args.log_name,
                "perm" + str(args.perm_id),
            )
        ):
            os.makedirs(
                os.path.join(
                    args.tb_dir,
                    args.dataset,
                    args.joint_da_loss,
                    str(args.class_num) + "class",
                    str(args.shot_num) + "shot",
                    args.cl_aug,
                    args.log_name,
                    "perm" + str(args.perm_id),
                )
            )
        if not os.path.exists(
            os.path.join(
                args.log_dir,
                args.dataset,
                args.joint_da_loss,
                str(args.class_num) + "class",
                str(args.shot_num) + "shot",
                args.cl_aug,
                args.log_name,
                "perm" + str(args.perm_id),
            )
        ):
            os.makedirs(
                os.path.join(
                    args.log_dir,
                    args.dataset,
                    args.joint_da_loss,
                    str(args.class_num) + "class",
                    str(args.shot_num) + "shot",
                    args.cl_aug,
                    args.log_name,
                    "perm" + str(args.perm_id),
                )
            )
        writer = SummaryWriter(
            os.path.join(
                args.tb_dir,
                args.dataset,
                args.joint_da_loss,
                str(args.class_num) + "class",
                str(args.shot_num) + "shot",
                args.cl_aug,
                args.log_name,
                "perm" + str(args.perm_id),
                cur_time,
            )
        )
        fh = logging.FileHandler(
            os.path.join(
                args.log_dir,
                args.dataset,
                args.joint_da_loss,
                str(args.class_num) + "class",
                str(args.shot_num) + "shot",
                args.cl_aug,
                args.log_name,
                "perm" + str(args.perm_id),
                cur_time + ".log",
            ),
            mode="a",
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    for arg in vars(args):
        logger.info("{}={}".format(arg.upper(), getattr(args, arg)))
    logger.info("")
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")  # type: ignore
    streams = collect_from_json(args.dataset, args.stream_root, "stream")
    label2idx = {0: 0}
    for st in streams:
        for lb in st:
            if lb not in label2idx:
                label2idx[lb] = len(label2idx)
    streams_indexed = [[label2idx[l] for l in st] for st in streams]
    model = BertED(args.class_num + 1, args.input_map)  # define model
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        # lr=args.lr,
        lr=0.01,
        weight_decay=args.decay,
        eps=args.adamw_eps,
        betas=(0.9, 0.999),
    )  # TODO: Hyper parameters
    if args.parallel == "DDP":
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=local_rank, world_size=args.world_size)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    elif args.parallel == "DP":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
        model = nn.DataParallel(
            model, device_ids=[int(it) for it in args.device_ids.split(" ")]
        )
    criterion_ce = nn.CrossEntropyLoss()
    criterion_fd = nn.CosineEmbeddingLoss()
    all_labels = []
    all_labels = list(
        set([t for stream in streams_indexed for t in stream if t not in all_labels])
    )
    task_idx = [i for i in range(len(streams_indexed))]
    labels = all_labels.copy()
    learned_types = [0]
    prev_learned_types = [0]
    dev_scores_ls = []
    exemplars = Exemplars()  # TODO:
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        task_idx = task_idx[state_dict["stage"] :]
        # TODO: test use
        labels = state_dict["labels"]
        learned_types = state_dict["learned_types"]
        prev_learned_types = state_dict["prev_learned_types"]
    if args.early_stop:
        e_pth = "./outputs/early_stop/" + args.log_name + ".pth"
    for stage in task_idx:
        logger.info(f"Stage {stage}")
        logger.info(f"Loading train instances for stage {stage}")
        if args.single_label:
            stream_dataset = collect_sldataset(
                args.dataset, args.data_root, "train", label2idx, stage, streams[stage]
            )
        else:
            stream_dataset = collect_dataset(
                args.dataset,
                args.data_root,
                "train",
                label2idx,
                stage,
                [i for item in streams[stage:] for i in item],
            )
        if args.parallel == "DDP":
            stream_sampler = DistributedSampler(stream_dataset, shuffle=True)
            org_loader = DataLoader(
                dataset=stream_dataset,
                sampler=stream_sampler,
                batch_size=args.batch_size,
                collate_fn=lambda x: x,
            )
        else:
            org_loader = DataLoader(
                dataset=stream_dataset,
                shuffle=True,
                batch_size=args.batch_size,
                collate_fn=lambda x: x,
            )
        stage_loader = org_loader
        if stage > 0:
            if args.early_stop and no_better == args.patience:
                logger.info("Early stopping finished, loading stage: " + str(stage))
                model.load_state_dict(torch.load(e_pth))
            prev_model = deepcopy(model)  # TODO:test use
            for item in streams_indexed[stage - 1]:
                if not item in prev_learned_types:
                    prev_learned_types.append(item)
            logger.info(
                f"Loading train instances without negative instances for stage {stage}"
            )
            exemplar_dataset = collect_exemplar_dataset(
                args.dataset,
                args.data_root,
                "train",
                label2idx,
                stage - 1,
                streams[stage - 1],
            )
            exemplar_loader = DataLoader(
                dataset=exemplar_dataset,
                batch_size=64,
                shuffle=True,
                collate_fn=lambda x: x,
            )
            exemplars.set_exemplars(
                prev_model, exemplar_loader, len(learned_types), device
            )
            if not args.no_replay:
                stage_loader = exemplars.build_stage_loader(stream_dataset)
            if args.rep_aug != "none":

                e_loader = exemplars.build_stage_loader(MAVEN_Dataset([], [], [], []))

        for item in streams_indexed[stage]:
            if not item in learned_types:
                learned_types.append(item)
        logger.info(f"Learned types: {learned_types}")
        logger.info(f"Previous learned types: {prev_learned_types}")
        dev_score = None
        no_better = 0
        # for name, param in model.named_parameters():
        #     print(f'Layer: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}')

        for ep in range(args.epochs):
            if stage == 0 and args.skip_first:
                continue
            logger.info("-" * 100)
            logger.info(f"Stage {stage}: Epoch {ep}")
            logger.info("Training process")
            model.train()
            logger.info("Training batch:")
            iter_cnt = 0
            for bt, batch in enumerate(tqdm(stage_loader)):
                iter_cnt += 1
                optimizer.zero_grad()
                train_x, train_y, train_masks, train_span = zip(*batch)
                train_x = torch.LongTensor(train_x).to(device)
                train_masks = torch.LongTensor(train_masks).to(device)
                train_y = [torch.LongTensor(item).to(device) for item in train_y]
                train_span = [torch.LongTensor(item).to(device) for item in train_span]
                # print("data x-----")
                # print(train_x)
                # print("data y-----")

                # print(train_y)
                # print("data span-----")

                # print(train_span)
                # print("data-----")

                # #

                #
                return_dict = model(train_x, train_masks, train_span)

                outputs, context_feat, trig_feat = (
                    return_dict["outputs"],
                    return_dict["context_feat"],
                    return_dict["trig_feat"],
                )
                # print("trigg------")
                # print(len(trig_feat))
                # print(trig_feat)
                # print("reps----")
                # print(len(return_dict["reps"]))
                # print(return_dict["reps"])

                # pi_star = compute_optimal_transport(D_W_P, D_T_P, C, epsilon=epsilon)
                # print(f"truoc khi mask {train_y}")
                # print(learned_types)
                for i in range(len(train_y)):
                    invalid_mask_label = torch.BoolTensor(
                        [item not in learned_types for item in train_y[i]]
                    ).to(device)
                    train_y[i].masked_fill_(invalid_mask_label, 0)

                # print('mask_label: ')
                # print(invalid_mask_label)
                # print(train_y)
                true_trig, true_label_onehot = get_true_y(train_y)

                p_wi_order = return_dict["p_wi_order"]
                p_tj = return_dict["p_tj"]
                D_W_P_order = return_dict['D_W_P_order']
                D_T_P = return_dict['D_T_P']
                # print(f'p_wi_order: {p_wi_order}')
                # print(f'true_trig: {true_trig}')
                loss_TI = compute_loss_TI(p_wi=p_wi_order,true_y=true_trig)
                # print(f'loss_TI {loss_TI}')
                # print(f'true_label_onehot: {true_label_onehot}')
                loss_TP = compute_loss_TP(p_tj=p_tj, true_label=true_label_onehot)

                last_hidden_state_order = return_dict['last_hidden_state_order']
                label_embeddings = model.get_label_embeddings()
                # print(f'size of last_hidden_state: {last_hidden_state}')
                # print(last_hidden_state)
                # cost_matrix = compute_cost_transport(last_hidden_state_order=last_hidden_state_order,label_embeddings=label_embeddings)
                # print(f'cost_matrix[0].requires_grad: {cost_matrix[0].requires_grad}')
                cost_matrix = return_dict['cost_matrix']
                # print(f'C size: {C.size()}')
                # pi_star = compute_optimal_transport_plane_for_batch(D_W_P_order=D_W_P_order,D_T_P=D_T_P,cost_matrix=cost_matrix)
                # print(f'size of pi_star: {pi_star.size()}')
                pi_star = return_dict['pi_star']

                L_task = compute_loss_Task_check_grad(pi_star_list=pi_star,y_true=train_y)
                # print('---in L_task---')
                print(f'L_task.requires_grad in main_copy: {L_task.requires_grad}')
                # print(f'pi_star.requires_grad: {pi_star.requires_grad}')
                # print(f'train_y.requires_grad: {train_y.requires_grad}')


                # print(f'L_task: {L_task}')
                # print(f'true_label size: {true_label.size()}')
                # Tính L_task: Negative Log-Likelihood Loss
                # print(f'train_y[2]: {train_y[2]}') 
                pi_g = get_pi_g(y_true=train_y)
                # print(f'pi_g[2]: {pi_g[2]}')
                # print(f'pi_star[2]: {pi_star[2]}')
                # print(f'cost[2]: {cost_matrix[2]}')
                Dist_pi_star = compute_Dist_pi_star(pi_star=pi_star,cost_matrix=cost_matrix)
                Dist_pi_g = compute_Dist_pi_g(pi_g=pi_g,cost_matrix=cost_matrix)
                L_OT = compute_loss_OT(Dist_pi_star,Dist_pi_g)
                alpha_task = 1.0
                alpha_OT = 0.01
                alpha_LT_I = 0.05
                alpha_LT_P = 0.01
                loss_ot = (
                    alpha_task * L_task
                    + alpha_OT * L_OT
                    + alpha_LT_I * loss_TI
                    + alpha_LT_P * loss_TP
                )

                print(f"task {L_task}")
                # print(f"OT: {L_OT}")
                # print(f"TI {loss_TI}")
                # print(f"TP {loss_TP}")
                # print(f"loss_ot {loss_ot}")

        #         loss, loss_ucl, loss_aug, loss_fd, loss_pd, loss_tlcl = 0, 0, 0, 0, 0, 0
        #         # ce_y = torch.cat(train_y)
        #         # ce_outputs = outputs
        #         if (args.ucl or args.tlcl) and (
        #             stage > 0 or (args.skip_first_cl != "ucl+tlcl" and stage == 0)
        #         ):
        #             reps = return_dict["reps"]
        #             bs, hdim = reps.shape
        #             aug_repeat_times = args.aug_repeat_times
        #             da_x = train_x.clone().repeat((aug_repeat_times, 1))
        #             da_y = train_y * aug_repeat_times
        #             da_masks = train_masks.repeat((aug_repeat_times, 1))
        #             da_span = train_span * aug_repeat_times
        #             tk_len = torch.count_nonzero(da_masks, dim=-1) - 2
        #             perm = [torch.randperm(item).to(device) + 1 for item in tk_len]
        #             if args.cl_aug == "shuffle":
        #                 for i in range(len(tk_len)):
        #                     da_span[i] = (
        #                         torch.where(
        #                             da_span[i].unsqueeze(2)
        #                             == perm[i].unsqueeze(0).unsqueeze(0)
        #                         )[2].view(-1, 2)
        #                         + 1
        #                     )
        #                     da_x[i, 1 : 1 + tk_len[i]] = da_x[i, perm[i]]
        #             elif args.cl_aug == "RTR":
        #                 rand_ratio = 0.25
        #                 rand_num = (rand_ratio * tk_len).int()
        #                 special_ids = [103, 102, 101, 100, 0]
        #                 all_ids = torch.arange(model.backbone.config.vocab_size).to(
        #                     device
        #                 )
        #                 special_token_mask = torch.ones(
        #                     model.backbone.config.vocab_size
        #                 ).to(device)
        #                 special_token_mask[special_ids] = 0
        #                 all_tokens = all_ids.index_select(
        #                     0, special_token_mask.nonzero().squeeze()
        #                 )
        #                 for i in range(len(rand_num)):
        #                     token_idx = torch.arange(tk_len[i]).to(device) + 1
        #                     trig_mask = torch.ones(token_idx.shape).to(device)
        #                     if args.dataset == "ACE":
        #                         span_pos = (
        #                             da_span[i][da_y[i].nonzero()].view(-1).unique() - 1
        #                         )
        #                     else:
        #                         span_pos = da_span[i].view(-1).unique() - 1
        #                     trig_mask[span_pos] = 0
        #                     token_idx_ntrig = token_idx.index_select(
        #                         0, trig_mask.nonzero().squeeze()
        #                     )
        #                     replace_perm = torch.randperm(token_idx_ntrig.shape.numel())
        #                     replace_idx = token_idx_ntrig[replace_perm][: rand_num[i]]
        #                     new_tkn_idx = torch.randperm(len(all_tokens))[: rand_num[i]]
        #                     da_x[i, replace_idx] = all_tokens[new_tkn_idx].to(device)
        #             # if args.dataset == "ACE":
        #             #     da_return_dict = model(da_x, da_masks)
        #             # else:
        #             da_return_dict = model(da_x, da_masks, da_span)
        #             da_outputs, da_reps, da_context_feat, da_trig_feat = (
        #                 da_return_dict["outputs"],
        #                 da_return_dict["reps"],
        #                 da_return_dict["context_feat"],
        #                 da_return_dict["trig_feat"],
        #             )

        #             if args.ucl:
        #                 if not (
        #                     (
        #                         args.skip_first_cl == "ucl"
        #                         or args.skip_first_cl == "ucl+tlcl"
        #                     )
        #                     and stage == 0
        #                 ):
        #                     ucl_reps = torch.cat([reps, da_reps])
        #                     ucl_reps = normalize(ucl_reps, dim=-1)
        #                     Adj_mask_ucl = torch.zeros(
        #                         bs * (1 + aug_repeat_times), bs * (1 + aug_repeat_times)
        #                     ).to(device)
        #                     for i in range(aug_repeat_times):
        #                         Adj_mask_ucl += torch.eye(
        #                             bs * (1 + aug_repeat_times)
        #                         ).to(device)
        #                         Adj_mask_ucl = torch.roll(Adj_mask_ucl, bs, -1)
        #                     loss_ucl = compute_CLLoss(
        #                         Adj_mask_ucl, ucl_reps, bs * (1 + aug_repeat_times)
        #                     )
        #             if args.tlcl:
        #                 if not (
        #                     (
        #                         args.skip_first_cl == "tlcl"
        #                         or args.skip_first_cl == "ucl+tlcl"
        #                     )
        #                     and stage == 0
        #                 ):
        #                     tlcl_feature = torch.cat([trig_feat, da_trig_feat])
        #                     tlcl_feature = normalize(tlcl_feature, dim=-1)
        #                     tlcl_lbs = torch.cat(train_y + da_y)
        #                     mat_size = tlcl_feature.shape[0]
        #                     tlcl_lbs_oh = F.one_hot(tlcl_lbs).float()
        #                     Adj_mask_tlcl = torch.matmul(tlcl_lbs_oh, tlcl_lbs_oh.T)
        #                     Adj_mask_tlcl = Adj_mask_tlcl * (
        #                         torch.ones(mat_size) - torch.eye(mat_size)
        #                     ).to(device)
        #                     loss_tlcl = compute_CLLoss(
        #                         Adj_mask_tlcl, tlcl_feature, mat_size
        #                     )
        #             loss = loss + loss_ucl + loss_tlcl
        #         #             if args.joint_da_loss == "ce" or args.joint_da_loss == "mul":
        #         #                 ce_y = torch.cat(train_y + da_y)
        #         #                 ce_outputs = torch.cat([outputs, da_outputs])
        #         #         ce_outputs = ce_outputs[:, learned_types]
        #                 # loss_ce = criterion_ce(ce_outputs, ce_y)
        #         loss = loss + loss_ot
        #         w = len(prev_learned_types) / len(learned_types)

        #         if args.rep_aug != "none" and stage > 0:
        #             outputs_aug, aug_y = [], []
        #             for e_batch in e_loader:
        #                 exemplar_x, exemplars_y, exemplar_masks, exemplar_span = zip(
        #                     *e_batch
        #                 )
        #                 exemplar_radius = [exemplars.radius[y[0]] for y in exemplars_y]
        #                 exemplar_x = torch.LongTensor(exemplar_x).to(device)
        #                 exemplar_masks = torch.LongTensor(exemplar_masks).to(device)
        #                 exemplars_y = [
        #                     torch.LongTensor(item).to(device) for item in exemplars_y
        #                 ]
        #                 exemplar_span = [
        #                     torch.LongTensor(item).to(device) for item in exemplar_span
        #                 ]
        #                 if args.rep_aug == "relative":
        #                     aug_return_dict = model(
        #                         exemplar_x,
        #                         exemplar_masks,
        #                         exemplar_span,
        #                         torch.sqrt(torch.stack(exemplar_radius)).unsqueeze(-1),
        #                     )
        #                 else:
        #                     aug_return_dict = model(
        #                         exemplar_x,
        #                         exemplar_masks,
        #                         exemplar_span,
        #                         torch.sqrt(
        #                             torch.stack(list(exemplars.radius.values())).mean()
        #                         ),
        #                     )
        #                 output_aug = aug_return_dict["outputs_aug"]
        #                 outputs_aug.append(output_aug)
        #                 aug_y.extend(exemplars_y)
        #             outputs_aug = torch.cat(outputs_aug)
        #             if args.leave_zero:
        #                 outputs_aug[:, 0] = 0
        #             outputs_aug = outputs_aug[:, learned_types].squeeze(-1)
        #             loss_aug = criterion_ce(outputs_aug, torch.cat(aug_y))
        #             loss = args.gamma * loss + args.theta * loss_aug

        #         if stage > 0 and args.distill != "none":
        #             prev_model.eval()
        #             with torch.no_grad():
        #                 prev_return_dict = prev_model(train_x, train_masks, train_span)
        #                 prev_outputs, prev_feature = (
        #                     prev_return_dict["outputs"],
        #                     prev_return_dict["context_feat"],
        #                 )

        #                 if args.joint_da_loss == "dist" or args.joint_da_loss == "mul":
        #                     outputs = torch.cat([outputs, da_outputs])
        #                     context_feat = torch.cat([context_feat, da_context_feat])
        #                     prev_return_dict_cl = prev_model(da_x, da_masks, da_span)
        #                     prev_outputs_cl, prev_feature_cl = (
        #                         prev_return_dict_cl["outputs"],
        #                         prev_return_dict_cl["context_feat"],
        #                     )
        #                     prev_outputs, prev_feature = torch.cat(
        #                         [prev_outputs, prev_outputs_cl]
        #                     ), torch.cat([prev_feature, prev_feature_cl])
        #             prev_valid_mask_op = torch.nonzero(
        #                 torch.BoolTensor(
        #                     [
        #                         item in prev_learned_types
        #                         for item in range(args.class_num + 1)
        #                     ]
        #                 ).to(device)
        #             )
        #             if args.distill == "fd" or args.distill == "mul":
        #                 prev_feature = normalize(
        #                     prev_feature.view(-1, prev_feature.shape[-1]), dim=-1
        #                 )
        #                 cur_feature = normalize(
        #                     context_feat.view(-1, prev_feature.shape[-1]), dim=-1
        #                 )
        #                 loss_fd = criterion_fd(
        #                     prev_feature,
        #                     cur_feature,
        #                     torch.ones(prev_feature.size(0)).to(device),
        #                 )  # TODO: Don't know whether the code is right
        #             else:
        #                 loss_fd = 0
        #             if args.distill == "pd" or args.distill == "mul":
        #                 T = args.temperature
        #                 if args.leave_zero:
        #                     prev_outputs[:, 0] = 0
        #                 prev_outputs = prev_outputs[:, prev_valid_mask_op].squeeze(-1)
        #                 cur_outputs = outputs[:, prev_valid_mask_op].squeeze(-1)
        #                 prev_p = torch.softmax(prev_outputs / T, dim=-1)
        #                 p = torch.log_softmax(cur_outputs / T, dim=-1)
        #                 loss_pd = -torch.mean(torch.sum(prev_p * p, dim=-1), dim=0)
        #             else:
        #                 loss_pd = 0
        #             if args.dweight_loss and stage > 0:
        #                 loss = loss * (1 - w) + (loss_fd + loss_pd) * w
        #             else:
        #                 loss = loss + args.alpha * loss_fd + args.beta * loss_pd
                # L_task.requires_grad_ = True
                L_task.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                print(f'grad: {L_task.grad}')
                for i, pi_star_i in enumerate(pi_star):
                    print(f"Gradient for pi_star[{i}]:\n{pi_star_i.grad}")  
                print(f'grad: {L_task.grad}')
                for i, cost in enumerate(cost_matrix):
                    print(f"Gradient for cost[{i}]:\n{cost.grad}")  
                for i, dw in enumerate(D_W_P_order):
                    print(f"Gradient for dw[{i}]:\n{dw.grad}")  
                for i, dt in enumerate(D_T_P):
                    print(f"Gradient for dt[{i}]:\n{dt.grad}")  
                # L_task.backward()
                optimizer.step()


        #     logger.info(f"loss_ot: {loss_ot}")
        #     logger.info(f"loss_ucl: {loss_ucl}")
        #     logger.info(f"loss_tlcl: {loss_tlcl}")
        #     # logger.info(f'loss_ecl: {loss_ecl}')
        #     logger.info(f"loss_aug: {loss_aug}")
        #     logger.info(f"loss_fd: {loss_fd}")
        #     logger.info(f"loss_pd: {loss_pd}")
        #     logger.info(f"loss_all: {loss}")

        #     if ((ep + 1) % args.eval_freq == 0 and args.early_stop) or (ep + 1) == args.epochs: # TODO TODO
        #         # Evaluation process
        #         logger.info("Evaluation process")
        #         model.eval()
        #         with torch.no_grad():
        #             if args.single_label:
        #                 eval_dataset = collect_eval_sldataset(args.dataset, args.data_root, 'test', label2idx, None, [i for item in streams for i in item])
        #             else:
        #                 eval_dataset = collect_dataset(args.dataset, args.data_root, 'test', label2idx, None, [i for item in streams for i in item])
        #             eval_loader = DataLoader(
        #                 dataset=eval_dataset,
        #                 shuffle=False,
        #                 batch_size=4,
        #                 collate_fn=lambda x:x)
        #             calcs = Calculator()
        #             for batch in tqdm(eval_loader):
        #                 eval_x, eval_y, eval_masks, eval_span = zip(*batch)
        #                 eval_x = torch.LongTensor(eval_x).to(device)
        #                 eval_masks = torch.LongTensor(eval_masks).to(device)
        #                 eval_y = [torch.LongTensor(item).to(device) for item in eval_y]
        #                 eval_span = [torch.LongTensor(item).to(device) for item in eval_span]  
        #                 eval_return_dict = model(eval_x, eval_masks, eval_span)
        #                 eval_outputs = eval_return_dict['outputs']
        #                 valid_mask_eval_op = torch.BoolTensor([idx in learned_types for idx in range(args.class_num + 1)]).to(device)
        #                 for i in range(len(eval_y)):
        #                     invalid_mask_eval_label = torch.BoolTensor([item not in learned_types for item in eval_y[i]]).to(device)
        #                     eval_y[i].masked_fill_(invalid_mask_eval_label, 0)
                        
        #                 eval_D_W_P_order = eval_return_dict['D_W_P_order']
        #                 eval_D_T_P = eval_return_dict['D_T_P']
        #                 eval_last_hidden_state = eval_return_dict['last_hidden_state_order']
        #                 eval_label_embeddings = model.get_label_embeddings()
        #                 eval_cost_matrix = compute_cost_transport(eval_last_hidden_state,eval_label_embeddings)
        #                 eval_pi_star = compute_optimal_transport_plane_for_batch(eval_D_W_P_order,eval_D_T_P,eval_cost_matrix)
        #                 eval_y_pred = get_y_pred(eval_pi_star)


        #                 if args.leave_zero:
        #                     eval_outputs[:, 0] = 0
        #                 # eval_outputs = eval_outputs[:, valid_mask_eval_op].squeeze(-1)
        #                 calcs.extend(torch.cat(eval_y_pred), torch.cat(eval_y))
        #             bc, (precision, recall, micro_F1) = calcs.by_class(learned_types)
        #             if args.log:
        #                 writer.add_scalar(f'score/epoch/marco_F1', micro_F1,  ep + 1 + args.epochs * stage)
        #             if args.log and (ep + 1) == args.epochs:
        #                 writer.add_scalar(f'score/stage/marco_F1', micro_F1, stage)
        #             logger.info(f'marco F1 {micro_F1}')
        #             dev_scores_ls.append(micro_F1)
        #             logger.info(f"Dev scores list: {dev_scores_ls}")
        #             logger.info(f"bc:{bc}")
        #             if args.early_stop:
        #                 if dev_score is None or dev_score < micro_F1:
        #                     no_better = 0
        #                     dev_score = micro_F1
        #                     torch.save(model.state_dict(), e_pth)
        #                 else:
        #                     no_better += 1
        #                     logger.info(f'No better: {no_better}/{args.patience}')
        #                 if no_better >= args.patience:
        #                     logger.info("Early stopping with dev_score: " + str(dev_score))
        #                     if args.log:
        #                         writer.add_scalar(f'score/stage/marco_F1', micro_F1, stage)
        #                     break

        # for tp in streams_indexed[stage]:
        #     if not tp == 0:
        #         labels.pop(labels.index(tp))
        # save_stage = stage
        # if args.save_dir and local_rank == 0:
        #     state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'stage':stage + 1, 
        #                     'labels':labels, 'learned_types':learned_types, 'prev_learned_types':prev_learned_types}
        #     save_pth = os.path.join(args.save_dir, "perm" + str(args.perm_id))
        #     save_name = f"stage_{save_stage}_{cur_time}.pth"
        #     if not os.path.exists(save_pth):
        #         os.makedirs(save_pth)
        #     logger.info(f'state_dict saved to: {os.path.join(save_pth, save_name)}')
        #     torch.save(state, os.path.join(save_pth, save_name))
        #     os.remove(e_pth)


if __name__ == "__main__":
    args = parse_arguments()
    if args.parallel == "DDP":
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)
    else:
        train(0, args)
