import numpy
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import math
from typing import Any, Dict, Tuple, List, Union, Set
import warnings
from collections import OrderedDict
from sklearn.cluster import KMeans
#from pycave.bayes import GMM
from sklearn.mixture import GaussianMixture as GMM
from torch.nn.modules.linear import Linear
from torchmeta.modules import MetaLinear, MetaSequential, MetaModule
from transformers import AutoModelForMaskedLM
from tqdm import tqdm
import random as rd
import json
from collections import Counter
from huggingface_hub import hf_hub_download

rd.seed(0)
torch.manual_seed(0)
numpy.random.seed(0)

from utils.options import parse_arguments

opts = parse_arguments()
opts.lm_temp = 2
class CustomMetaSequential(nn.Sequential, MetaModule):
    __doc__ = nn.Sequential.__doc__

    def forward(self, input, params=None, return_all_layer=False):
        hidden = []
        for name, module in self._modules.items():
            if isinstance(module, MetaModule):
                input = module(input, params=self.get_subdict(params, name))
            elif isinstance(module, nn.Module):
                input = module(input)
            else:
                raise TypeError('The module must be either a torch module '
                                '(inheriting from `nn.Module`), or a `MetaModule`. '
                                'Got type: `{0}`'.format(type(module)))
            if return_all_layer:
                hidden.append(input)
        if return_all_layer:
            return input, torch.stack(hidden)
        return input
    
class dropout(nn.Dropout):
    
    def __init__(self, input_dim, p, device="cuda", fixed = False):
        super().__init__(p)
        self.fixed = fixed
        self.device = device
        self.p = p
        self.tau = 0.5
        self.max_alpha = 1.0
        self.log_alpha = nn.Parameter(torch.Tensor(input_dim))
        self.reset_parameters()
    
    def sample_gumbel(self, batch, size):
        g = torch.Tensor(batch, size).uniform_(0,1).type(torch.float).to(self.device)
        return -torch.log(-torch.log(g + 1e-20)+1e-20)
    
    def gumbel_softmax(self, theta, g1, g2):

        comp1 = (torch.log(theta) + g1)/self.tau
        comp2 = (torch.log(1- theta) + g2)/self.tau

        combined = torch.cat((comp1.unsqueeze(2), comp2.unsqueeze(2)), dim = 2) 
        max_comp = torch.max(combined, dim = 2)[0]
        pi_t = torch.exp(comp1 - max_comp)/ (torch.exp(comp2-max_comp) + torch.exp(comp1 - max_comp))

        return pi_t
    
    def reset_parameters(self):
        alpha = math.sqrt(self.p/(1-self.p))
        torch.nn.init.uniform_(self.log_alpha, -1, 0)
        #torch.nn.init.constant_(self.log_alpha, math.log(alpha))
        
    def forward(self, features):
        if self.training == False:
            return features
        
        #alpha = (1+alpha*noise)
        if self.fixed == False:
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max= 0)
            alpha = self.log_alpha.unsqueeze(0)
            alpha = alpha.repeat(features.size(0), 1)
            alpha = torch.exp(alpha)
            #noise = torch.randn(alpha.size()).to(self.device)
            alpha = self.gumbel_softmax(alpha, self.sample_gumbel(alpha.size(0), alpha.size(1)), self.sample_gumbel(alpha.size(0), alpha.size(1)))
            
            features = features*alpha#.unsqueeze(0)
        else:
            noise = torch.randn(features.size()).to(self.device)
            features = features*(noise*self.p+1)
        
        return features
    
import torch
import torch.nn.functional as F

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
        err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))
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

# Example usage:
# M, a, b, lambda_sh are your input tensors
# result = sinkhorn_pytorch(M, a, b, lambda_sh)
class lm_ot(torch.nn.Module):
    
    def __init__(self, max_slots, linear, device:str):
        super().__init__()
        if opts.llm2vec == False:
            vocab_size = 28996
            hidden_size = 1024
            lm_head = AutoModelForMaskedLM.from_pretrained("bert-large-cased").cls.to(device)
        else:
            vocab_size = 128256
            hidden_size = 4096
            lm_head = torch.nn.Linear(4096, 128256, bias=False)
            path = hf_hub_download(repo_id="vietdata/llama3_head", filename="lm_head.pth")
            lm_head.load_state_dict(torch.load(path))
        self.hidden_size = hidden_size
        self.model = lm_head
        self.linear = linear
        self.r2v = torch.nn.Linear(512, vocab_size)
        if opts.llm2vec == False:
            self.vocab = self.model.predictions.decoder.weight#train_head(self.model.predictions.decoder.weight)
        else:
            self.vocab = self.model.weight
        
        self.topics = torch.nn.ParameterList([torch.nn.Parameter(torch.nn.init.normal_(torch.empty(1,hidden_size), 0, 0.01)) for i in range(max_slots-1)])
        self.to(device)
        self.device = device
        with open("verb.json") as f:
            self.verbs = json.load(f)
        self.non_verbs = [i for i in range(vocab_size) if i not in self.verbs]
    
    def get_element(self, feature, indices): ## để lấy phần từ featrue (matrix) dọc theo trục indicase (chiều dọc)
        return feature[torch.arange(feature.size(0)).unsqueeze(1), indices.repeat(feature.size(0), 1)]

    def get_m(self, nslots, verbs): ## tính cosine giữa nslots với verbs, 1 - scores cho biết chi phí (cost)
        if opts.llm2vec == False:
            vocab = self.get_element(self.model.predictions.decoder.weight.data.T, verbs).T 
        else:
            vocab = self.get_element(self.model.weight.data.T, verbs).T 
        vocab = vocab/(torch.norm(vocab,dim=1, keepdim=True)+1e-8)
        topics = torch.cat([param for param in self.topics[:nslots].parameters()])
        topics = topics/(torch.norm(topics,dim=1, keepdim=True)+1e-8)
        scores = torch.matmul(topics, vocab.T)
        return 1-scores
    
    def ot_loss(self, topics, vocabs, nslots, verbs): 
        m = self.get_m(nslots-1, verbs)
        loss = 0
        bach_size = 1024
        for i in range(topics.size(0)//bach_size+1):
            topics_ = topics[i*bach_size: (i+1)*bach_size]
            vocabs_ = vocabs[i*bach_size: (i+1)*bach_size]
            if topics_.size(0) == 0:
                break
            loss += sinkhorn_pytorch(m, topics_.T, vocabs_.T , 20).mean()
        return loss
    
    def softmax(self, feat):
        return torch.nn.functional.softmax(feat, dim=-1)
    
    def forward2(self, inputs, outputs):
        output11 = self.softmax(self.model(inputs[:, :self.hidden_size])/opts.lm_temp)
        output12 = self.softmax(self.model(inputs[:, self.hidden_size:])/opts.lm_temp)
        output1 = (output11+output12)/2
        output2 = self.softmax(self.r2v(outputs))
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        input_ = F.log_softmax(output2, dim=1)
        target = F.softmax(output1, dim=1)
        return kl_loss(input_, target)
    
    def forward(self, inputs, outputs, nslots):
        if not opts.ot:
            return self.forward2(inputs, outputs)
        else:
            non_verbs = rd.sample(self.non_verbs, 3000//5)
            spans = list(set(self.spans + self.verbs))
            verbs = torch.tensor(spans, device=self.device).reshape(1,-1)#torch.cat([self.verbs, torch.tensor(spans, device=self.device).reshape(1, -1)], dim=1)
            output11 = self.softmax(self.get_element(self.model(inputs[:, :self.hidden_size]), verbs)/opts.lm_temp)
            output12 = self.softmax(self.get_element(self.model(inputs[:, self.hidden_size:]), verbs)/opts.lm_temp)
            output1 = (output11+output12)/2
            #output1 = self.get_element(output1, self.verbs)
            output1 = output1/torch.sum(output1, dim=1, keepdim=True)
            output2 = self.softmax(self.linear(outputs)[:, 1:nslots])
            
            return self.ot_loss(output2, output1, nslots, verbs)
    
    def get_prototype(self, features):
        scores = []
        batch_size = 1024
        for idx in range(features.size(0)//batch_size + 1):
            inputs = features[batch_size*idx:batch_size*(idx+1)]
            if inputs.size(0) == 0:
                break
            output11 = self.softmax(self.model(inputs[:, :self.hidden_size])/opts.lm_temp)
            output12 = self.softmax(self.model(inputs[:, self.hidden_size:])/opts.lm_temp)
            scores.append((output11+output12)/2)
        return torch.cat(scores)
    
    def expand(self, nslots):
        for i in range(nslots-1):
            if self.topics[i].requires_grad == True:
                self.topics[i].requires_grad = False

class LInEx(MetaModule):
    def __init__(self, input_dim: int, hidden_dim: int, max_slots: int, init_slots: int,dropout_type="adap", p:int=0.1, 
                 device: Union[torch.device, None] = None, **kwargs) -> None:
        super().__init__()

        if input_dim != hidden_dim:
            if dropout_type != "normal":
                self.is_adap = True if dropout_type == "adap" else False
                self.input_map = CustomMetaSequential(OrderedDict({
                    "linear_0": MetaLinear(input_dim, hidden_dim),
                    "relu_0": nn.ReLU(),
                    "dropout_0": dropout(hidden_dim, device=device, p=p, fixed=True if dropout_type != "adap" else False),
                    "linear_1": MetaLinear(hidden_dim, hidden_dim),
                    "relu_1": nn.ReLU(),
                }))
            else:
                self.is_adap = False
                self.input_map = CustomMetaSequential(OrderedDict({
                    "linear_0": MetaLinear(input_dim, hidden_dim),
                    "relu_0": nn.ReLU(),
                    "dropout_0": nn.Dropout(p),
                    "linear_1": MetaLinear(hidden_dim, hidden_dim),
                    "relu_1": nn.ReLU(),
                }))
        else:
            self.input_map = lambda x: x
        self.classes = MetaLinear(hidden_dim, max_slots, bias=False)
        self.lm_head = lm_ot(max_slots, self.classes, device)
        self.lm_mode = "ot"
        for param in self.lm_head.model.parameters():
            param.requires_grad = False
        self.learned_classes =  None
        _mask = torch.zeros(1, max_slots, dtype=torch.float, device=device)
        _mask[:, init_slots:] = float("-inf")
        self.register_buffer(name="_mask", tensor=_mask)
        self.crit = nn.CrossEntropyLoss()
        self.device = device
        self.to(device=device)
        self.nslots = init_slots
        self.max_slots = max_slots
        self.maml = True
        self.outputs = {}
        self.history = None
        self.exemplar_features = None
        self.exemplar_labels = None
        self.dev_exemplar_features = None
        self.dev_exemplar_labels = None
        self.gen = None
        self.gmms = {}
        self.gen_features = []
        self.gen_labels = []
        self.trained_replay = set()
        self.trained_generate = set()
        #self.alpha = torch.nn.Parameter(torch.normal(torch.zeros(input_dim), torch.ones(input_dim)*-1)).to(self.device)
        with open("data/MAVEN/streams.json") as f:
            task2id = json.load(f)
            id2task = {}
            start = 1
            for i in range(len(task2id)):
                task2id[i].remove(0)
                for j in range(start, start+len(task2id[i])):
                    id2task[j] = 0
                start += len(task2id[i])
            id2task[0] = 0
        self.id2task = id2task

    def set_class(self):
        if self.learned_classes is None:
            self.learned_classes = self.classes.weight.data[:self.nslots].detach().clone()
            self.learned_classes.requires_grad = False
        else:
            previous = self.learned_classes.size(0)
            self.learned_classes = torch.cat([self.learned_classes, self.classes.weight.data[previous:self.nslots]], dim=0)
            self.learned_classes.requires_grad = False

    def class_loss(self):
        if self.learned_classes is None:
            return 0
        previous = self.learned_classes.size(0)
        current_classes = self.classes.weight
        learned_classes = torch.cat([self.learned_classes.detach().clone(), current_classes.data[previous:].detach().clone()], dim=0)
        loss = ((learned_classes - current_classes)**2+1e-8).sum(axis=1)**(1/2)
        #print(loss.mean())
        #print("current", current_classes)
        #loss.mean().backward()
        #print(self.classes.weight.grad)
        return loss.mean()

    @property
    def mask(self, ):
        self._mask[:, :self.nslots] = 0
        self._mask[:, self.nslots:] = float("-inf")
        return self._mask

    def idx_mask(self, idx: Union[torch.LongTensor, int, List[int], None] = None,
                 max_idx: Union[torch.LongTensor, int, None] = None):
        assert (idx is not None) or (max_idx is not None)
        assert (idx is None) or (max_idx is None)
        mask = torch.zeros_like(self._mask) + float("-inf")
        if idx is not None:
            mask[:, idx] = 0
        if max_idx is not None:
            if isinstance(max_idx, torch.LongTensor):
                max_idx = max_idx.item()
            mask[:, :max_idx] = 0
        return mask

    @property
    def features(self):
        return self.classes.weight[:self.nslots]
    
    def label_loss(self, label, labels, features, tau):

        idx = (labels == label).detach()
        pidx = torch.nonzero(idx, as_tuple=True)[0].tolist()
        nidx = torch.nonzero((idx == False).detach(), as_tuple=True)[0].tolist()

        if label == 0:
            nidx = list(range(features.size()[0]))
        pfeatures = features[pidx, :]
        nfeatures = features[nidx, :]
        sims = torch.sum(torch.exp(torch.matmul(pfeatures, nfeatures.T)/tau), dim=-1)
        if label != 0:
            pidx = pidx[::-1]
        rpfeatures = features[pidx, :]
        psims = torch.exp(torch.sum(rpfeatures*pfeatures, dim=1)/tau)
        if label != 0:
            cross = torch.sum(-torch.log(psims/(psims+sims)))
        else:
            cross = torch.sum(-torch.log(psims/(sims)))
            if cross == 0:
                print("zero", psims, sims)
                print("features", features.size())
                print("pn", pfeatures.size(), nfeatures.size())
        if opts.debug:
            if cross < 0:
                import pdb
                pdb.set_trace()
            
        return cross / len(pidx)
    
    def contrastive_loss(self, features, labels, tau=0.1):
        # if opts.debug:
            # import pdb
            # pdb.set_trace()
        features = features/(torch.sum(features**2, dim=-1, keepdims=True)**(1/2)+1e-5)
        loss = 0
        for i, label in enumerate(set(labels.cpu().numpy().tolist())):
            loss += self.label_loss(label, labels, features, tau)
        """if loss == 0:
            print(labels)"""
        return loss     

    def compute_KL_bernoulli(self, theta1):
        """
        theta: is a probability which make variable get value 1 in Bernoulli distribution, theta belongs to [0,1]

        Return KL divergence between two Bernoulli distribution
        """
        theta1 = torch.exp(theta1)
        theta2 = torch.ones(theta1.size(), device=self.device)*0.8
        return (theta1*torch.log(theta1/theta2) + (1-theta1)*torch.log((1-theta1+1e-6)/(1-theta2))).sum()

    def forward(self, batch, nslots: int = -1, contrastive:bool=False, return_loss_list: bool=False, generate: bool = False, sample_size: int = 5, exemplar: bool = False,
                exemplar_distill: bool = False, feature_distill: bool = False, mul_distill=False, distill: bool = False,
                return_loss: bool = True, return_feature: bool = False, tau: float = 1.0, log_outputs: bool = True,
                params=None, hyer_distill: bool = False, return_all_features: bool = False, init:int=0.1, lambda_coef:int=1e-1):
        balance_na = self.balance_na
        loss_list = []
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            features, labels = batch
        else:
            features, labels = batch.features, batch.labels
            spans = batch.spans.cpu().numpy()
            spans = list(set([j for i in spans for j in i]))
            self.lm_head.spans = spans

        all_inputs = []
        all_labels = []
        tasks = []

        inputs = self.input_map(features, params=self.get_subdict(params, "input_map"),
                                             return_all_layer=False)
        scores = self.classes(inputs, params=self.get_subdict(params, "classes"))
        all_inputs.append(inputs)
        all_labels.append(labels)
        if torch.any(torch.isnan(scores)):
            print(scores[0])
            print("input", inputs)
            input('a')
        if nslots == -1:
            scores += self.mask
            if torch.any(torch.isnan(scores)):
                print(scores[0])
                input()
            nslots = self.nslots
        else:
            scores += self.idx_mask(max_idx=nslots)
        scores[:, 0] = 0
        if scores.size(0) != labels.size(0):
            import pdb
            pdb.set_trace()
            assert scores.size(0) % labels.size(0) == 0
            labels = labels.repeat_interleave(scores.size(0) // labels.size(0), dim=0)
        else:
            labels = labels
        if log_outputs:
            pred = torch.argmax(scores, dim=1)
            acc = torch.mean((pred == labels).float())
            self.outputs["accuracy"] = acc.item()
            self.outputs["prediction"] = pred.detach().cpu()
            self.outputs["label"] = labels.clone().detach().cpu()
            self.outputs["input_features"] = features.clone().detach().cpu()
            self.outputs["encoded_features"] = inputs.clone().detach().cpu()

        if return_loss:
            labels.masked_fill_(labels >= nslots, 0)
            plabels = torch.nonzero(labels > 0, as_tuple=True)[0].tolist()
            nlabels = torch.nonzero(labels==0, as_tuple=True)[0].tolist()
            valid = labels < nslots
            nvalid = torch.sum(valid.float())
            if nvalid == 0:
                loss = 0
            else:
                loss = self.crit(scores[valid], labels[valid])
                if torch.isnan(loss):
                    print(labels, nslots, scores[:, :nslots])
                    input()
            loss = 0
            loss += 0.2*self.class_loss()
            loss += 0.2*self.lm_head(features, inputs, self.nslots)
            if len(plabels) > 0:
                loss1 = self.crit(scores[plabels, :], labels[plabels])
                #loss_list.append(loss1*5)
                loss += loss1*balance_na*len(plabels)/(len(plabels)*balance_na + len(nlabels))
            if len(nlabels) > 0:
                loss2 = self.crit(scores[nlabels, :], labels[nlabels])
                #loss_list.append(loss2)
                loss += loss2*len(nlabels)/(len(plabels)*balance_na + len(nlabels))
            rate = 128/self.loader_length
            try:
                loss_list.append((loss+lambda_coef*self.compute_KL_bernoulli(self.input_map[2].log_alpha)*rate)
                                 if self.is_adap else loss)
            except:
                import pdb
                pdb.set_trace()
            
            if distill and self.history is not None:
                if hyer_distill:
                    old_scores, old_inputs, old_hiddens_mlp = self.forward(batch, nslots=self.history["nslots"],
                                                                           return_loss=False, log_outputs=False,
                                                                           return_feature=True,
                                                                           params=self.history["params"],
                                                                           return_all_features=True)
                    old_hiddens_mlp = old_hiddens_mlp.detach()
                else:
                    old_scores, old_inputs = self.forward(batch, nslots=self.history["nslots"], return_loss=False,
                                                          log_outputs=False, return_feature=True,
                                                          params=self.history["params"])
                old_scores = old_scores.detach()
                old_inputs = old_inputs.detach()
                new_scores = scores[:, :self.history["nslots"]]
                if mul_distill:
                    loss_distill = - torch.sum(torch.softmax(old_scores * tau, dim=1)
                                               * torch.log_softmax(new_scores * tau, dim=1), dim=1).mean()
                    old_dist = torch.softmax(old_scores / tau, dim=1)
                    old_valid = (old_dist[:, 0] < 0.9)
                    old_num = torch.sum(old_valid.float())
                    if old_num > 0:
                        loss_mul_distill = - torch.sum(old_dist[old_valid]
                                                       * torch.log_softmax(new_scores[old_valid], dim=1), dim=1).sum()
                        loss_distill = (loss_distill * old_dist.size(0) + loss_mul_distill) \
                                       / (old_dist.size(0) + old_num)
                else:
                    loss_distill = - torch.sum(torch.softmax(old_scores * tau, dim=1)
                                               * torch.log_softmax(new_scores * tau, dim=1), dim=1).mean()

                if hyer_distill:

                    try:
                        for i in range(1, len(old_hiddens_mlp), 3):
                            loss_f_distill_t = (1 - (
                                    old_hiddens_mlp[i] / old_hiddens_mlp[i].norm(dim=-1, keepdim=True) * hiddens_mlp[i]
                                    / hiddens_mlp[i].norm(dim=-1,
                                                          keepdim=True)).sum(
                                dim=-1)).mean(dim=0)
                            loss_distill += loss_f_distill_t

                        # huydq59 -- score output distill
                        loss_scores_distill = (1 - (old_scores / old_scores.norm(dim=-1, keepdim=True) *
                                           new_scores / new_scores.norm(dim=-1, keepdim=True)).sum(dim=-1)).mean(dim=0)
                        loss_distill += loss_scores_distill
                        
                    except:
                        print('hyer distill fail')
                        import pdb
                        pdb.set_trace()

                # just distill output of mlp
                elif feature_distill:
                    loss_f_distill = (1 - (old_inputs / old_inputs.norm(dim=-1, keepdim=True) *
                                           inputs / inputs.norm(dim=-1, keepdim=True)).sum(dim=-1)).mean(dim=0)
                    loss_distill += loss_f_distill
                loss_list.append(loss_distill)
                d_weight = self.history["nslots"]
                c_weight = (self.nslots - self.history["nslots"])
                loss = (d_weight * loss_distill + c_weight * loss) / (d_weight + c_weight)
                if torch.isnan(loss):
                    print(old_scores, new_scores)
                    input()
            if exemplar and self.exemplar_features is not None:
                exemplar_features = self.exemplar_features
                exemplar_labels = self.exemplar_labels

                nidx = self.sample_data(exemplar_labels, self.trained_replay, k=sample_size)
                exemplar_features = exemplar_features[nidx]
                exemplar_labels = exemplar_labels[nidx]
                tasks = []
                

                if opts.debug:
                    # import pdb
                    # pdb.set_trace()
                    print(len(self.gen_labels))
                    print(len(exemplar_labels))

                if generate:
                    if len(self.gen_features) > 0:
                        nidx = self.sample_data(self.gen_labels, self.trained_generate, k=sample_size)
                        gen_features = [self.gen_features[feat] for feat in nidx]
                        gen_labels = [self.gen_labels[feat] for feat in nidx]
                        gen_features = torch.cat(gen_features, dim=0)
                        gen_labels = torch.tensor(gen_labels)  # , device=self.device)
                        exemplar_features = torch.cat((exemplar_features, gen_features), dim=0)
                        exemplar_labels = torch.cat((exemplar_labels, gen_labels))
                        
                if exemplar_features.size(0) < 128:
                    exemplar_inputs = self.input_map(exemplar_features.to(self.device),
                                                     params=self.get_subdict(params, "input_map"))
                    exemplar_scores = self.classes(exemplar_inputs, params=self.get_subdict(params, "classes"))
                else:
                    exemplar_scores = []
                    exemplar_inputs = []
                    for _beg in range(0, exemplar_features.size(0), 128):
                        _features = exemplar_features[_beg:_beg + 128, :]
                        _inputs = self.input_map(_features.to(self.device),
                                                 params=self.get_subdict(params, "input_map"))
                        exemplar_inputs.append(_inputs)
                        exemplar_scores.append(self.classes(_inputs, params=self.get_subdict(params, "classes")))
                    exemplar_inputs = torch.cat(exemplar_inputs, dim=0)
                    exemplar_scores = torch.cat(exemplar_scores, dim=0)
                all_inputs.append(exemplar_inputs)
                all_labels.append(exemplar_labels.to(self.device))
                exemplar_scores[:, 0] = 0.
                loss_exemplar = self.crit(exemplar_scores + self.mask, exemplar_labels.to(self.device))
                if self.is_adap:
                    rate = exemplar_features.size(0)/(len(self.gen_labels) + len(self.exemplar_labels))
                    loss_exemplar += lambda_coef*self.compute_KL_bernoulli(self.input_map[2].log_alpha)*rate
                loss_list.append(loss_exemplar)
                if torch.isnan(loss_exemplar):
                    print(exemplar_labels, nslots)
                    input()
                if exemplar_distill:
                    if exemplar_features.size(0) < 128:
                        if hyer_distill:
                            exemplar_old_inputs, exemplar_old_hiddens_mlp = self.input_map(
                                exemplar_features.to(self.device),
                                params=self.get_subdict(self.history["params"], "input_map"),
                                return_all_layer=True)
                        else:
                            exemplar_old_inputs = self.input_map(
                                exemplar_features.to(self.device),
                                params=self.get_subdict(self.history["params"], "input_map"))

                        exemplar_old_scores = self.classes(exemplar_old_inputs,
                                                           params=self.get_subdict(self.history["params"], "classes"))
                    else:
                        exemplar_old_scores = []
                        exemplar_old_inputs = []
                        if hyer_distill:
                            exemplar_old_hiddens_mlp = []

                        for _beg in range(0, exemplar_features.size(0), 128):
                            _features = exemplar_features[_beg:_beg + 128, :]
                            if hyer_distill:
                                _inputs, _exemplar_old_hiddens_mlp = self.input_map(_features.to(self.device),
                                                                                    params=self.get_subdict(
                                                                                        self.history["params"],
                                                                                        "input_map"),
                                                                                    return_all_layer=True)
                                exemplar_old_hiddens_mlp.append(_exemplar_old_hiddens_mlp)
                            else:
                                _inputs = self.input_map(_features.to(self.device),
                                                         params=self.get_subdict(self.history["params"], "input_map"))

                            exemplar_old_inputs.append(_inputs)
                            exemplar_old_scores.append(
                                self.classes(_inputs, params=self.get_subdict(self.history["params"], "classes")))

                        exemplar_old_inputs = torch.cat(exemplar_old_inputs, dim=0)
                        exemplar_old_scores = torch.cat(exemplar_old_scores, dim=0) 

                        if hyer_distill:
                            exemplar_old_hiddens_mlp = torch.cat(exemplar_old_hiddens_mlp, dim=1)

                    exemplar_old_scores[:, 0] = 0.
                    exemplar_old_scores = exemplar_old_scores[:self.history["nslots"]]
                    loss_exemplar_distill = - torch.sum(
                        torch.softmax(exemplar_old_scores[:self.history["nslots"]] * tau, dim=1) * torch.log_softmax(
                            exemplar_scores[:self.history["nslots"]], dim=1), dim=1).mean()
                    #loss_list.append(self.lm_head(exemplar_features.to(self.device), exemplar_inputs.to(self.device), self.nslots))
                    loss_exemplar_distill += 0.2*self.lm_head(exemplar_features.to(self.device), exemplar_inputs.to(self.device), self.nslots)

                    if hyer_distill:
                        # exemplar_old_hiddens_mlp

                        for i in range(1, len(exemplar_old_hiddens_mlp), 3):
                            try:
                                loss_exemplar_feat_distill_t = (1 - (
                                        exemplar_old_hiddens_mlp[i] / exemplar_old_hiddens_mlp[i].norm(dim=-1,
                                                                                                       keepdim=True) *
                                        exemplar_inputs[i]
                                        / exemplar_inputs[i].norm(dim=-1,
                                                                  keepdim=True)).sum(
                                    dim=-1)).mean(dim=0)
                                loss_exemplar_distill += loss_exemplar_feat_distill_t
                            except:
                                print('exemplare hyer distill fail')
                                import pdb
                                pdb.set_trace()

                        # huydq59 - exemplar score output distill
                        loss_exemplar_scores_distill = (1 - (
                                    exemplar_old_scores[:self.history["nslots"]] / exemplar_old_scores[
                                                                                   :self.history[
                                                                                       "nslots"]].norm(dim=-1,
                                                                                                       keepdim=True) *
                                    exemplar_scores[:self.history["nslots"]] / exemplar_scores[
                                                                               :self.history["nslots"]].norm(
                                dim=-1, keepdim=True)).sum(dim=-1)).mean(dim=0)
                        loss_exemplar_distill += loss_exemplar_scores_distill

                    elif feature_distill:
                        loss_exemplar_feat_distill = (1 - (exemplar_old_inputs / exemplar_old_inputs.norm(dim=-1,
                                                                                                          keepdim=True) * exemplar_inputs / exemplar_inputs.norm(
                            dim=-1, keepdim=True)).sum(dim=-1)).mean(dim=0)
                        loss_exemplar_distill += loss_exemplar_feat_distill

                    d_weight = self.history["nslots"]
                    c_weight = (self.nslots - self.history["nslots"])
                    loss_list.append(loss_exemplar_distill)
                    loss_exemplar = (d_weight * loss_exemplar_distill + c_weight * loss_exemplar) / (
                                d_weight + c_weight)
                e_weight = exemplar_features.size(0)
                loss = (nvalid * loss + e_weight * loss_exemplar) / (nvalid + e_weight)
                if torch.isnan(loss):
                    print(loss, loss_exemplar)
            if return_loss_list == True:
                if contrastive == True:
                    inputs = torch.cat(all_inputs, dim=0)
                    labels = torch.cat(all_labels, dim=0)
                    labels.masked_fill_(labels >= nslots, 0)
                    if inputs.size()[0] > 1 and len(set(labels.cpu().numpy().tolist())) > 1:
                        if labels[0] == 0:
                            idx = list(range(labels.size()[0]))
                        else:
                            idx = self.sample_data(labels, trained=set(), k=1)
                        loss_list.insert(0, self.contrastive_loss(inputs[idx, :], labels[idx]))
                return loss_list
            if contrastive == True:
                inputs = torch.cat(all_inputs, dim=0)
                labels = torch.cat(all_labels, dim=0)
                labels.masked_fill_(labels >= nslots, 0)
                if inputs.size()[0] > 1 and len(set(labels.cpu().numpy().tolist())) > 1:
                    if labels[0] == 0:
                        idx = list(range(labels.size()[0]))
                    else:
                        idx = self.sample_data(labels, trained=set(), k=1)
                    w = inputs.size()[0]/(inputs.size()[0]+len(idx))
                    loss =  w*loss + (1-w)*self.contrastive_loss(inputs[idx, :], labels[idx])
            return loss
        else:
            if return_all_features:
                return scores[:, :nslots], inputs, hiddens_mlp
            elif return_feature:
                return scores[:, :nslots], inputs
            else:
                return scores[:, :nslots]

    
    def sample_data(self, labels, trained, k: int = 5):
        nidx = []
        s = 0
        if len(trained) == len(labels):
            trained.clear()
        for e in range(len(labels)):
            sampled = []
            if labels[e] != labels[s]:
                set_labels = [i for i in range(s, e) if i not in trained]
                if len(set_labels) == 0:
                    s = e
                    continue
                if len(set_labels) < k:
                    set_labels = set_labels * (1 + k // len(set_labels))
                sampled = rd.sample(set_labels, k=min(k, len(set_labels)))
                s = e
                trained.update(set(sampled))
                nidx.extend(sampled)
            if e == len(labels) - 1:
                set_labels = [i for i in range(s, e + 1) if i not in trained]
                if len(set_labels) == 0:
                    s = e
                    continue
                if len(set_labels) < k:
                    set_labels = set_labels * (1 + k // len(set_labels))
                sampled = rd.sample(set_labels, k=min(k, len(set_labels)))
                trained.update(set(sampled))
                nidx.extend(sampled)
        return nidx
    
    def score(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def clone_params(self, ):
        return OrderedDict({k: v.clone().detach() for k, v in self.meta_named_parameters()})

    def set_history(self, ):
        self.history = {"params": self.clone_params(), "nslots": self.nslots}

    def set_exemplar(self, dataloader, generate: bool = False, generate_ratio:int=20, center_ratio:int=1, mode: str = "herding", num_clusters: int = 10, 
                     q: int = 20, params=None, label_sets: Union[List, Set, None] = None, collect_none: bool = False,
                     use_input: bool = False, output_only: bool = False, output: Union[str, None] = None):
        self.eval()
        with torch.no_grad():
            ifeat = [];
            ofeat = [];
            label = []
            num_batches = len(dataloader)
            for batch in tqdm(dataloader, "collecting exemplar", ncols=128):
                batch = batch.to(self.device)
                loss = self.forward(batch, params=params)
                ifeat.append(self.outputs["input_features"])
                if use_input:
                    ofeat.append(self.outputs["input_features"])
                else:
                    ofeat.append(self.outputs["encoded_features"])
                label.append(self.outputs["label"])
            ifeat = torch.cat(ifeat, dim=0)
            ofeat = torch.cat(ofeat, dim=0)
            label = torch.cat(label, dim=0)
            label_ = label
            nslots = max(self.nslots, torch.max(label).item() + 1)
            exemplar = {}
            if label_sets is None:
                if collect_none:
                    label_sets = range(nslots)
                else:
                    label_sets = range(1, nslots)
            else:
                if collect_none:
                    if 0 not in label_sets:
                        label_sets = sorted([0] + list(label_sets))
                    else:
                        label_sets = sorted(list(label_sets))
                else:
                    label_sets = sorted([t for t in label_sets if t != 0])
            for i in label_sets:
                idx = (label == i)
                if i == 0:
                    # random sample for none type
                    nidx = torch.nonzero(idx, as_tuple=True)[0].tolist()
                    exemplar[i] = numpy.random.choice(nidx, q, replace=False).tolist()
                    continue
                if torch.any(idx):
                    exemplar[i] = []
                    nidx = torch.nonzero(idx, as_tuple=True)[0].tolist()
                    mfeat = torch.mean(ofeat[idx], dim=0, keepdims=True)
                    if len(nidx) < q:
                        exemplar[i].extend(nidx * (q // len(nidx)) + nidx[:(q % len(nidx))])
                    else:
                        if mode in  ["kmeans", "GMM"]:
                            if mode == "kmeans":
                                cluster_model = KMeans(n_clusters=num_clusters).fit(ofeat[nidx])
                                clusters = torch.tensor(cluster_model.cluster_centers_, device=self.device)
                            else:
                                cluster_model = GMM(n_components=num_clusters).fit(ofeat[nidx])
                                clusters = torch.tensor(cluster_model.means_, device=self.device)
                            pred_labels = torch.tensor(cluster_model.predict(ofeat[nidx]), device=self.device)

                            # huydq59
                            # bigger cluster produce more sample in replay set
                            try:
                                stats = Counter(list(set(cluster_model.predict(ofeat[nidx]))))
                                cluster_to_number = []
                                for lb in stats:
                                    temp_n = int(math.floor(stats[lb] * 1.0 / len(ofeat[nidx]) * q))
                                    temp_n = temp_n if temp_n else temp_n + 1
                                    cluster_to_number.append(temp_n)
                                cluster_to_number[-1] = q - sum(cluster_to_number[:-1])
                                iter_t = len(cluster_to_number) - 1
                                #
                                while cluster_to_number[-1] <= 0:
                                    iter_t -= 1
                                    if cluster_to_number[iter_t] <= 1:
                                        continue
                                    cluster_to_number[iter_t] -= 1
                                    cluster_to_number[-1] += 1
                                    if iter_t == 0:
                                        iter_t = len(cluster_to_number) - 1
                            except Exception as e:
                                print(e)
                                print('kmeans calculate number error. Input to continue')

                            num = len(nidx)
                            sample_idx = []
                            for index_cluster, cluster in enumerate(clusters):
                                # huydq
                                # for j in range(q//num_clusters):
                                # while draw_count < cluster_to_number[index_cluster]:
                                idx = (pred_labels == index_cluster)
                                label_nidx = torch.tensor(nidx, device=self.device)[idx].tolist()
                                sample = cluster.to(self.device).unsqueeze(0)
                                for _ in range(cluster_to_number[index_cluster]):
                                    if len(label_nidx) == 0:
                                        break
                                    cluster.to(self.device)
                                    if (center_ratio !=0) and sample.size()[0] <= max(1,math.floor(cluster_to_number[index_cluster]*center_ratio)):
                                        dfeat = torch.sum((ofeat[label_nidx].to(self.device) - cluster) ** 2, dim=1)
                                        tfeat = torch.argmin(dfeat)
                                        sample_idx.append(label_nidx[tfeat])
                                        sample = torch.cat([sample,  ofeat[label_nidx[tfeat]].to(self.device).unsqueeze(0)], dim=0)
                                        label_nidx.pop(tfeat.item())
                                        #dists = torch.sum((ofeat[label_nidx].unsqueeze(1).to(self.device) - sample.unsqueeze(0))**2, dim=-1)
                                        #dists = torch.sum(dists, dim=1)
                                        #tfeat = torch.argmin(dists)
                                        #sample_idx.append(label_nidx[tfeat])
                                        #sample = torch.cat([sample,  ofeat[label_nidx[tfeat]].unsqueeze(0).to(self.device)], dim=0)
                                        #label_nidx.pop(tfeat.item())
                                    else:
                                        dists = torch.sum((ofeat[label_nidx].unsqueeze(1).to(self.device) - sample.unsqueeze(0))**2, dim=-1)
                                        dists = torch.sum(dists, dim=1)
                                        tfeat = torch.argmax(dists)
                                        sample_idx.append(label_nidx[tfeat])
                                        sample = torch.cat([sample,  ofeat[label_nidx[tfeat]].unsqueeze(0).to(self.device)], dim=0)
                                        label_nidx.pop(tfeat.item())

                            exemplar[i].extend(sample_idx)
                        elif mode == "herding":
                            for j in range(q):
                                if j == 0:
                                    dfeat = torch.sum((ofeat[nidx] - mfeat) ** 2, dim=1)
                                else:
                                    cfeat = ofeat[exemplar[i]].sum(dim=0, keepdims=True)
                                    cnum = len(exemplar[i])
                                    dfeat = torch.sum((mfeat * (cnum + 1) - ofeat[nidx] - cfeat) ** 2, )
                                tfeat = torch.argmin(dfeat)
                                exemplar[i].append(nidx[tfeat])
                                nidx.pop(tfeat.item())
            exemplar = {i: ifeat[v] for i, v in exemplar.items()}
            exemplar_features = []
            exemplar_labels = []
            for label, features in exemplar.items():
                exemplar_features.append(features)
                exemplar_labels.extend([label] * features.size(0))
            exemplar_features = torch.cat(exemplar_features, dim=0).cpu()
            exemplar_labels = torch.LongTensor(exemplar_labels).cpu()
            if not output_only or output is not None:
                if output == "train" or output is None:
                    if self.exemplar_features is None:
                        self.exemplar_features = exemplar_features
                        self.exemplar_labels = exemplar_labels
                    else:
                        self.exemplar_features = torch.cat((self.exemplar_features, exemplar_features), dim=0)
                        self.exemplar_labels = torch.cat((self.exemplar_labels, exemplar_labels), dim=0)
                elif output == "dev":
                    if self.dev_exemplar_features is None:
                        self.dev_exemplar_features = exemplar_features
                        self.dev_exemplar_labels = exemplar_labels
                    else:
                        self.dev_exemplar_features = torch.cat((self.dev_exemplar_features, exemplar_features), dim=0)
                        self.dev_exemplar_labels = torch.cat((self.dev_exemplar_labels, exemplar_labels), dim=0)
            print(self.exemplar_features.size())
        if generate:
            for i in tqdm(label_sets):
                idx = (label_ == i)
                if torch.any(idx):
                    ifeats = ifeat[idx]
                    """
                    with open(f"log/features_by_label/{i}.json", "w") as f:
                        for ft in ifeats:
                            f.write(json.dumps(ft.cpu().numpy().tolist()) + "\n")
                    """
                    if len(ifeats) <= 2:
                        continue
                    try:
                        gmm = GMM(n_components=min(num_clusters, len(ifeats)) if len(ifeats) < 600 else min(num_clusters*2, len(ifeats)) , random_state=0, covariance_type='diag')
                        gmm.fit(ifeats)
                        self.gmms[i] = gmm                   
                    except Exception as e:
                        print(e)
                
            self.gen_features = []
            self.gen_labels = []       
            for i, gmm in tqdm(self.gmms.items()):
                nfeats = gmm.sample(generate_ratio*q)[0].tolist()
                nfeats = torch.tensor(nfeats)
                self.gen_features.extend(torch.split(nfeats,1))
                self.gen_labels.extend([i]*generate_ratio*q)
        return {i: v.cpu() for i, v in exemplar.items()}


    def initialize(self, exemplar, ninstances: Dict[int, int], gamma: float = 1.0, tau: float = 1.0, alpha: float = 0.5,
                   params=None):
        self.eval()

        with torch.no_grad():
            weight_norm = torch.norm(self.classes.weight[1:self.nslots], dim=1).mean(dim=0)
            label_inits = []
            label_kt = {}
            for label, feats in exemplar.items():
                exemplar_inputs = self.input_map(feats.to(self.device), params=self.get_subdict(params, "input_map"))
                exemplar_scores = self.classes(exemplar_inputs, params=self.get_subdict(params, "classes"))
                exemplar_scores = exemplar_scores + self.mask
                exemplar_scores[:, 0] = 0
                exemplar_weights = torch.softmax(exemplar_scores * tau, dim=1)
                normalized_inputs = exemplar_inputs / torch.norm(exemplar_inputs, dim=1, keepdim=True) * weight_norm
                proto = (exemplar_weights[:, :1] * normalized_inputs).mean(dim=0)
                knowledge = torch.matmul(exemplar_weights[:, 1:self.nslots], self.classes.weight[1:self.nslots]).mean(
                    dim=0)
                gate = alpha * math.exp(- ninstances[label] * gamma)
                # gate = 1 / (1 + ninstances[label] * gamma)
                rnd = torch.randn_like(proto) * weight_norm / math.sqrt(self.classes.weight.size(1))
                initvec = proto * gate + knowledge * gate + (1 - gate) * rnd
                label_inits.append((label, initvec.cpu()))
                label_kt[label] = exemplar_weights.mean(dim=0).cpu()
            label_inits.sort(key=lambda t: t[0])
            inits = []
            for i, (label, init) in enumerate(label_inits):
                assert label == self.nslots + i
                inits.append(init)
            inits = torch.stack(inits, dim=0)
            self.outputs["new2old"] = label_kt
        return inits.detach()

    def initialize2(self, exemplar, ninstances: Dict[int, int], gamma: float = 1.0, tau: float = 1.0,
                    alpha: float = 0.5, delta: float = 0.5, params=None):
        self.eval()

        def top_p(probs, p=0.9):
            _val, _idx = torch.sort(probs, descending=True, dim=1)
            top_mask = torch.zeros_like(probs).float() - float("inf")
            for _type in range(probs.size(0)):
                accumulated = 0
                _n = 0
                while accumulated < p or _n <= 1:
                    top_mask[_type, _idx[_type, _n]] = 0
                    accumulated += _val[_type, _n]
                    _n += 1
            return top_mask

        with torch.no_grad():
            weight_norm = torch.norm(self.classes.weight[1:self.nslots], dim=1).mean(dim=0)
            label_inits = []
            label_kt = {}
            for label, feats in exemplar.items():
                exemplar_inputs = self.input_map(feats.to(self.device), params=self.get_subdict(params, "input_map"))
                exemplar_scores = self.classes(exemplar_inputs, params=self.get_subdict(params, "classes"))
                exemplar_scores = exemplar_scores + self.mask
                exemplar_scores[:, 0] = 0
                top_mask = top_p(torch.softmax(exemplar_scores, dim=1))
                exemplar_scores = exemplar_scores + top_mask
                exemplar_scores[:, 0] = 0
                exemplar_weights = torch.softmax(exemplar_scores * tau, dim=1)
                normalized_inputs = exemplar_inputs / torch.norm(exemplar_inputs, dim=1, keepdim=True) * weight_norm
                proto = delta * (exemplar_weights[:, :1] * normalized_inputs).mean(dim=0)
                kweight = (1 - exemplar_weights[:, :1])
                knowledge = torch.matmul(
                    (1 - delta * exemplar_weights[:, :1]) * (exemplar_weights[:, 1:self.nslots] + 1e-8) / torch.clamp(
                        1 - exemplar_weights[:, :1], 1e-8), self.classes.weight[1:self.nslots]).mean(dim=0)
                gate = alpha * math.exp(- ninstances[label] * gamma)
                rnd = torch.randn_like(proto) * weight_norm / math.sqrt(self.classes.weight.size(1))
                initvec = proto * gate + knowledge * gate + (1 - gate) * rnd
                if torch.any(torch.isnan(initvec)):
                    print(proto, knowledge, rnd, gate, exemplar_weights[:, :1], exemplar_scores[-1, :self.nslots])
                    input()
                label_inits.append((label, initvec.cpu()))
                label_kt[label] = exemplar_weights.mean(dim=0).cpu()
            label_inits.sort(key=lambda t: t[0])
            inits = []
            for i, (label, init) in enumerate(label_inits):
                assert label == self.nslots + i
                inits.append(init)
            inits = torch.stack(inits, dim=0)
            self.outputs["new2old"] = label_kt
        return inits.detach()

    def set(self, features: torch.tensor, ids: Union[int, torch.Tensor, List, None] = None, max_id: int = -1):
        with torch.no_grad():
            if isinstance(ids, (torch.Tensor, list)):
                if torch.any(ids > self.nslots):
                    warnings.warn(
                        "Setting features to new classes. Using 'extend' or 'append' is preferred for new classes")
                self.classes.weight[ids] = features
            elif isinstance(ids, int):
                self.classes.weight[ids] = features
            else:
                if max_id == -1:
                    raise ValueError(f"Need input for either ids or max_id")
                self.classes.weight[:max_id] = features

    def append(self, feature):
        with torch.no_grad():
            self.classes.weight[self.nslots] = feature
            self.nslots += 1

    def extend(self, features):
        with torch.no_grad():
            features = features.to(self.device)
            if len(features.size()) == 1:
                warnings.warn("Extending 1-dim feature vector. Using 'append' instead is preferred.")
                self.append(features)
            else:
                nclasses = features.size(0)
                self.classes.weight[self.nslots:self.nslots + nclasses] = features
                self.nslots += nclasses


class BIC(LInEx):
    def __init__(self, input_dim: int, hidden_dim: int, max_slots: int, init_slots: int,
                 device: Union[torch.device, None] = None, **kwargs) -> None:
        super().__init__(input_dim, hidden_dim, max_slots, init_slots, device, **kwargs)
        self.correction_weight = nn.Parameter(torch.ones(1, dtype=torch.float, device=self.device, requires_grad=True))
        self.correction_bias = nn.Parameter(torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=True))
        self.correction_stream = [init_slots]

    def add_stream(self, num_classes):
        self.correction_stream.append(self.correction_stream[-1] + num_classes)

    def forward(self, batch, nslots: int = -1, bias_correction: str = "none", exemplar: bool = False,
                exemplar_distill: bool = False, distill: bool = False, return_loss: bool = True, tau: float = 1.0,
                log_outputs: bool = True, params=None):
        assert bias_correction in ["none", "last", "current"]
        if distill:
            assert bias_correction != "current"
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            features, labels = batch
        else:
            features, labels = batch.features, batch.labels
        inputs = self.input_map(features, params=self.get_subdict(params, "input_map"))
        scores = self.classes(inputs, params=self.get_subdict(params, "classes"))
        if nslots == -1:
            scores += self.mask
            nslots = self.nslots
        else:
            scores += self.idx_mask(max_idx=nslots)
        scores[:, 0] = 0
        if bias_correction == "current":
            assert len(self.correction_stream) >= 2
            scores[:, self.correction_stream[-2]:self.correction_stream[-1]] *= self.correction_weight
            scores[:, self.correction_stream[-2]:self.correction_stream[-1]] += self.correction_bias
        if scores.size(0) != labels.size(0):
            assert scores.size(0) % labels.size(0) == 0
            labels = labels.repeat_interleave(scores.size(0) // labels.size(0), dim=0)
        else:
            labels = labels
        if log_outputs:
            pred = torch.argmax(scores, dim=1)
            acc = torch.mean((pred == labels).float())
            self.outputs["accuracy"] = acc.item()
            self.outputs["prediction"] = pred.detach().cpu()
            self.outputs["label"] = labels.detach().cpu()
            self.outputs["input_features"] = features.detach().cpu()
            self.outputs["encoded_features"] = inputs.detach().cpu()
        if return_loss:
            labels.masked_fill_(labels >= nslots, 0)
            valid = labels < nslots
            nvalid = torch.sum(valid.float())
            if nvalid == 0:
                loss = 0
            else:
                loss = self.crit(scores[valid], labels[valid])
            if distill and self.history is not None:
                old_scores = self.forward(batch, nslots=self.history["nslots"], return_loss=False, log_outputs=False,
                                          params=self.history["params"]).detach()
                if bias_correction == "last":
                    old_scores[:, self.correction_stream[-2]:self.correction_stream[-1]] *= self.history[
                        'correction_weight']
                    old_scores[:, self.correction_stream[-2]:self.correction_stream[-1]] += self.history[
                        'correction_bias']
                new_scores = scores[:, :self.history["nslots"]]
                loss_distill = - torch.sum(
                    torch.softmax(old_scores * tau, dim=1) * torch.log_softmax(new_scores * tau, dim=1), dim=1).mean()
                d_weight = self.history["nslots"]
                c_weight = (self.nslots - self.history["nslots"])
                loss = (d_weight * loss_distill + c_weight * loss) / (d_weight + c_weight)
            if exemplar and self.exemplar_features is not None:
                if self.exemplar_features.size(0) < 128:
                    exemplar_inputs = self.input_map(self.exemplar_features.to(self.device),
                                                     params=self.get_subdict(params, "input_map"))
                    exemplar_scores = self.classes(exemplar_inputs, params=self.get_subdict(params, "classes"))
                else:
                    exemplar_scores = []
                    for _beg in range(0, self.exemplar_features.size(0), 128):
                        _features = self.exemplar_features[_beg:_beg + 128, :]
                        _inputs = self.input_map(_features.to(self.device),
                                                 params=self.get_subdict(params, "input_map"))
                        exemplar_scores.append(self.classes(_inputs, params=self.get_subdict(params, "classes")))
                    exemplar_scores = torch.cat(exemplar_scores, dim=0)
                exemplar_scores[:, 0] = 0.
                loss_exemplar = self.crit(exemplar_scores + self.mask, self.exemplar_labels.to(self.device))
                if exemplar_distill:
                    if self.exemplar_features.size(0) < 128:
                        exemplar_old_inputs = self.input_map(self.exemplar_features.to(self.device),
                                                             params=self.get_subdict(self.history["params"],
                                                                                     "input_map"))
                        exemplar_old_scores = self.classes(exemplar_old_inputs,
                                                           params=self.get_subdict(self.history["params"], "classes"))
                    else:
                        exemplar_old_scores = []
                        for _beg in range(0, self.exemplar_features.size(0), 128):
                            _features = self.exemplar_features[_beg:_beg + 128, :]
                            _inputs = self.input_map(_features.to(self.device),
                                                     params=self.get_subdict(self.history["params"], "input_map"))
                            exemplar_old_scores.append(
                                self.classes(_inputs, params=self.get_subdict(self.history["params"], "classes")))
                        exemplar_old_scores = torch.cat(exemplar_old_scores, dim=0)
                    exemplar_old_scores[:, 0] = 0.
                    if bias_correction == "last":
                        exemplar_old_scores[:, self.correction_stream[-2]:self.correction_stream[-1]] *= self.history[
                            'correction_weight']
                        exemplar_old_scores[:, self.correction_stream[-2]:self.correction_stream[-1]] += self.history[
                            'correction_bias']
                    exemplar_old_scores = exemplar_old_scores[:self.history["nslots"]]
                    loss_exemplar_distill = - torch.sum(
                        torch.softmax(exemplar_old_scores[:self.history["nslots"]] * tau, dim=1) * torch.log_softmax(
                            exemplar_scores[:self.history["nslots"]], dim=1), dim=1).mean()
                    d_weight = self.history["nslots"]
                    c_weight = (self.nslots - self.history["nslots"])
                    loss_exemplar = (d_weight * loss_exemplar_distill + c_weight * loss_exemplar) / (
                                d_weight + c_weight)
                e_weight = self.exemplar_features.size(0)
                loss = (nvalid * loss + e_weight * loss_exemplar) / (nvalid + e_weight)
                if torch.isnan(loss):
                    print(loss, loss_exemplar)
            return loss
        else:
            return scores[:, :nslots]

    def forward_correction(self, *args, **kwargs):
        '''
        training:
            entropy: normal
            distill:
                old, last
                Fold, Fold * correction_weight + correction_bias,
        '''
        if len(args) >= 3:
            args[2] = "current"
        else:
            kwargs["bias_correction"] = "current"
        return self.forward(*args, **kwargs)

    def set_history(self):
        super().set_history()
        self.history["correction_weight"] = self.correction_weight.item()
        self.history["correction_bias"] = self.correction_bias.item()

    def score(self, *args, **kwargs):
        if len(self.correction_stream) >= 2:
            return self.forward_correction(*args, **kwargs)
        else:
            if len(args) >= 3:
                args[2] = "none"
            else:
                kwargs["bias_correction"] = "none"
            return self.forward(*args, **kwargs)


class ICARL(LInEx):
    def __init__(self, input_dim: int, hidden_dim: int, max_slots: int, init_slots: int,
                 device: Union[torch.device, None] = None, **kwargs) -> None:
        super().__init__(input_dim, hidden_dim, max_slots, init_slots, device, **kwargs)
        self.none_feat = None

    def set_none_feat(self, dataloader, params=None):
        self.eval()
        with torch.no_grad():
            ifeat = [];
            ofeat = [];
            label = []
            num_batches = len(dataloader)
            for batch in tqdm(dataloader, "collecting exemplar"):
                batch = batch.to(self.device)
                loss = self.forward(batch, params=params)
                ifeat.append(self.outputs["input_features"])
                ofeat.append(self.outputs["encoded_features"])
                label.append(self.outputs["label"])
            ifeat = torch.cat(ifeat, dim=0)
            ofeat = torch.cat(ofeat, dim=0)
            label = torch.cat(label, dim=0)
            nslots = max(self.nslots, torch.max(label).item() + 1)
            exemplar = {}
            idx = (label == 0)
            self.none_feat = ofeat[idx].mean(dim=0).cpu()
            return self.none_feat

    def score(self, batch, exemplar=None, params=None):
        if exemplar is None:
            exemplar_labels, exemplar_features = self.exemplar_labels, self.exemplar_features
        else:
            exemplar_labels, exemplar_features = exemplar

        inputs = self.input_map(batch.features, params=self.get_subdict(params, "input_map"))
        scores = []
        scores.append(- torch.sum((inputs - self.none_feat.to(inputs.device).unsqueeze(0)) ** 2, dim=1))
        for i in range(1, self.nslots):
            label_idx = (exemplar_labels == i)
            label_features = exemplar_features[label_idx]
            label_inputs = self.input_map(label_features.to(inputs.device),
                                          params=self.get_subdict(params, "input_map")).mean(dim=0, keepdim=True)
            scores.append(- torch.sum((inputs - label_inputs) ** 2, dim=1))
        scores = torch.stack(scores, dim=0).transpose(0, 1)
        labels = batch.labels
        if scores.size(0) != labels.size(0):
            assert scores.size(0) % labels.size(0) == 0
            labels = labels.repeat_interleave(scores.size(0) // labels.size(0), dim=0)
        pred = torch.argmax(scores, dim=1)
        acc = torch.mean((pred == labels).float())
        labels.masked_fill_(labels >= self.nslots, 0)
        valid = labels < self.nslots
        nvalid = torch.sum(valid.float())
        if nvalid == 0:
            loss = 0
        else:
            loss = self.crit(scores[valid], labels[valid])
        self.outputs["accuracy"] = acc.item()
        self.outputs["prediction"] = pred.detach().cpu()
        self.outputs["label"] = labels.detach().cpu()
        self.outputs["input_features"] = batch.features.detach().cpu()
        self.outputs["encoded_features"] = inputs.detach().cpu()
        return loss


def test():  # sanity check
    m = LInEx(nhead=8, nlayers=3, hidden_dim=512, input_dim=2048, max_slots=30, init_slots=9,
              device=torch.device("cpu"))


if __name__ == "__main__":
    test()