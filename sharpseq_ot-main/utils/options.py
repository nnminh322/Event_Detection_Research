import argparse
import os
import glob

def define_arguments(parser):
    parser.add_argument('--json-root', type=str, default="./data", help="")
    parser.add_argument('--feature-root', type=str, default="data/features", help="")
    parser.add_argument('--stream-file', type=str, default="data/MAVEN/streams.json", help="")
    parser.add_argument('--batch-size', type=int, default=32, help="")
    parser.add_argument('--init-slots', type=int, default=13, help="")
    parser.add_argument('--patience', type=int, default=10, help="")
    parser.add_argument('--input-dim', type=int, default=2048, help="")
    parser.add_argument('--hidden-dim', type=int, default=512, help="")
    parser.add_argument('--max-slots', type=int, default=169, help="")
    parser.add_argument('--perm-id', type=int, default=0, help="")
    parser.add_argument('--no-gpu', action="store_true", help="don't use gpu")
    parser.add_argument('--gpu', type=int, default=0, help="gpu")
    parser.add_argument('--learning-rate', type=float, default=1e-4, help="")
    parser.add_argument('--decay', type=float, default=1e-2, help="")
    parser.add_argument('--kt-alpha', type=float, default=0.25, help="")
    parser.add_argument('--kt-gamma', type=float, default=0.05, help="")
    parser.add_argument('--kt-tau', type=float, default=1.0, help="")
    parser.add_argument('--kt-delta', type=float, default=0.5, help="")
    parser.add_argument('--seed', type=int, default=2147483647, help="random seed")
    parser.add_argument('--save-model', type=str, default="model", help="path to save checkpoints")
    parser.add_argument('--load-model', type=str, default="", help="path to saved checkpoint")
    parser.add_argument('--log-dir', type=str, default="./log/", help="path to save log file")
    parser.add_argument('--train-epoch', type=int, default=50, help='epochs to train')
    parser.add_argument('--test-only', action="store_true", help='is testing')
    parser.add_argument('--kt', action="store_true", help='')
    parser.add_argument('--kt2', action="store_true", help='')
    parser.add_argument('--finetune', action="store_true", help='')
    parser.add_argument('--load-first', type=str, default="", help="path to saved checkpoint")
    parser.add_argument('--skip-first', action="store_true", help='')
    parser.add_argument('--load-second', type=str, default="", help="path to saved checkpoint")
    parser.add_argument('--skip-second', action="store_true", help='')
    parser.add_argument('--balance', choices=['icarl', 'eeil', 'bic', 'none', 'fd', 'mul', 'nod'], default="none")
    parser.add_argument('--setting', choices=['classic', "new"], default="classic")
    parser.add_argument('--mode', choices=["kmeans", "herding", "GMM"], type=str, default="herding", help='exemplar algorithm')
    parser.add_argument('--kt_mode', choices=["kmeans", "herding", "GMM"], type=str, default="herding", help='KT')
    parser.add_argument('--clusters', type=int, default=4, help='the number of clusters')
    parser.add_argument('--generate',  action="store_true", help="")
    parser.add_argument('--sample-size', type=int, default=2, help="the sample size of each label in the replay and generated sets")
    parser.add_argument('--features-distill',  action="store_true",  help="whether do feature distillation (just distill span mlp output) or not")
    parser.add_argument('--hyer-distill',  action="store_true",  help="whether do feature hyer-distillation or not")
    parser.add_argument('--reduce-na',  action="store_true",  help="reduce number of negative instances")


    parser.add_argument('--new-test-mode',  action="store_true",  help="") # still debunging, not used
    parser.add_argument('--num_loss', type=int, default=4, help='epochs to train')
    parser.add_argument('--mul_task', action="store_true", help='epochs to train')
    parser.add_argument("--contrastive", action="store_true", help="contrastive loss")
    parser.add_argument("--mul_distill", action="store_true")
    parser.add_argument("--mul_task_type", type=str, choices=['NashMTL','PCGrad','IMTLG', 'MGDA'], default='NashMTL')
    parser.add_argument("--naive_replay", action="store_true")


    parser.add_argument("--debug", action="store_true", help="for debug")
    parser.add_argument("--colab_viet", action="store_true", help="util for run on colab")

    parser.add_argument('--datasetname',  type=str, choices=['MAVEN', 'ACE', 'ACE_lifelong'], default='MAVEN')
    parser.add_argument("--center-ratio", type=int, default=1, help="The number points that near to the center")
    parser.add_argument("--generate_ratio", type=int, default=20, help="The ratio between replay set and generated set")
    parser.add_argument("--naloss_ratio", type=int, default=4, help="")
    parser.add_argument("--dropout", type=str, choices=["normal", "adap", "fixed"], default="normal")
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--num_sam_loss", type=int, default=2)
    parser.add_argument("--sam", type=int, default=1, help="sam")
    parser.add_argument("--ot", action="store_true", help="for debug")
    parser.add_argument("--llm2vec", action="store_true", help="llm2vec")



PERM = [[0, 1, 2, 3,4], [4, 3, 2, 1, 0], [0, 3, 1, 4, 2], [1, 2, 0, 3, 4], [3, 4, 0, 1, 2]]

import json
def parse_arguments():
    parser = argparse.ArgumentParser()
    define_arguments(parser)
    args = parser.parse_args()
    args.log = os.path.join(args.log_dir, "logfile.log")

    if (not args.test_only) and os.path.exists(args.log_dir):
        existing_logs = glob.glob(os.path.join(args.log_dir, "*"))
        for _t in existing_logs:
            if 'exp.log' not in _t and not _t.endswith('.py'):
                os.remove(_t)
    print('Dump name space')
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(f'{args.log_dir}/options.json', 'w') as f:
        print(f'{args.log_dir}/options.json')
        json.dump(vars(args), f, ensure_ascii=False)

    return args