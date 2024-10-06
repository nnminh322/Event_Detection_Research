import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel
from configs import parse_arguments

args = parse_arguments()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")


class BertED(nn.Module):
    def __init__(self, class_num=args.class_num + 1, input_map=False):
        super().__init__()
        self.backbone = BertModel.from_pretrained(args.backbone)
        if not args.no_freeze_bert:
            print("Freeze Bert Parameters")
            for _, param in self.backbone.named_parameters():
                param.requires_grad = False
        else:
            print("Update Parameters")

        self.is_input_maping = input_map
        self.input_dim = self.backbone.config.hidden_size
        self.fc = nn.Linear(self.input_dim, class_num)
        if self.is_input_maping:
            self.map_hidden_dim = 512
            self.map_input_dim = self.input_dim * 2
            self.input_map = nn.Sequential(
                nn.Linear(self.map_input_dim, self.map_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.map_hidden_dim, self.map_hidden_dim),
                nn.ReLU(),
            )
            self.fc = nn.Linear(self.map_hidden_dim, class_num)
        # if flag is_input_maping is True, model will apply more deeper layer to understand context

    def forward(self, x, mask, span=None, aug=None):
        # x is the input data representing token indices with shape [batch_size, max_seq_len]
        # batch_size is the number of sentences in the batch
        # max_seq_len is the maximum number of tokens in a sentence, where padding may be applied
        # mask is a tensor of shape [batch_size, max_seq_len] that indicates which tokens are valid
        # (1 for valid tokens, 0 for padding tokens)
        # span is a tensor of shape [batch_size, num_spans, 2], where:
        # span [batch_size, max_seq_len, 2] is mark where is start trigger, end trigger and 2 is type of trigger
        # aug is an optional parameter for augmentation
        return_dict = {}
        backbone_output = self.backbone(x, attention_mask=mask)
        feature, pooled_feat = backbone_output[0], backbone_output[1]
        # feature [batch_size, seq_len, hidden_dim], cpooled_featls [batch_size, hidden_dim] is reps of sentence (is cls feed into an BertPool)
        context_feature = feature.view(-1, feature.shape[-1])
        cls = feature[:, 0, :].clone()
        # context_feature is context of all sentence in a batch, [batch_size x seq_len, hidden_dim]
        # simple is flatten and concat all token_presentation in all sentence
        if span != None:
            logits_of_pair_trigger_feature, pair_trigger_feature = [], []
            for i in range(len(span)):
                if self.is_input_maping:
                    x_cdt = torch.stack(
                        [
                            torch.index_select(feature[i], 0, span[i][:, j])
                            for j in range(span[i].size(-1))
                        ]
                    )
                    # for each sentence, x_cdt is stack of trigger 0 and trigger 1, [2, num_trigger, hidden_dim]
                    x_cdt.contiguous(1, 0, 2)
                    # -> [num_trigger,2,hidden_dim]
                    x_cdt.contiguous().view(x_cdt.size(0), x_cdt.size(-1) * 2)
                    # concate start and end trigger -> [num_trigger, 2*hidden_dim]
                    output_pair_trigger = self.input_map(x_cdt)
                else:
                    output_pair_trigger = torch.index_select(
                        feature[i], 0, span[i][:, 0]
                    ) + torch.index_select(feature[i], 0, span[i][:, 1])
                    # instead of concate start and end trigger then feed into deeper layer, this case simple is plus start and end trigger
                pair_trigger_feature.append(output_pair_trigger)
            pair_trigger_feature = torch.cat(pair_trigger_feature)
            # pair_trigger_feature [num_total_trigger, map_hidden_dim] (if apply deeper) or [batch_size, num_total trigger, hidden_dim]
            # pair_trigger_feature is reps of all trigger in all sentence in each batch
            # num_total_trigger is sum of trigger in all sentence (some sentence has more one trigger)
            logits_of_pair_trigger_feature = self.fc(pair_trigger_feature)
            # distribution of pair_trigger for labels [num_total_trigger, class_num]
            return_dict["context_feature"] = (
                context_feature  # context of all sentence in a batch
            )
            return_dict["outputs"] = (
                logits_of_pair_trigger_feature  # distribution of pair_trigger for labels
            )
            return_dict["trigger_feat"] = (
                pair_trigger_feature  # reps of all trigger in all sentence in each batch
            )
            return_dict["reps"] = cls  # cls of all sentence in each batch

            if aug is not None:
                pair_trigger_feature_aug = (
                    pair_trigger_feature + torch.rand_like(pair_trigger_feature) * aug
                )
                logits_of_pair_trigger_feature_aug = self.fc(pair_trigger_feature_aug)
                return_dict["trigger_feat_aug"] = pair_trigger_feature_aug
                return_dict["outputs_aug"] = logits_of_pair_trigger_feature_aug

            return return_dict

    def forward_backbone(self, x, masks):
        x = self.backbone(x, attention_mask=masks)
        x = x.last_hidden_state
        return x

    def forward_input_map(self, x):
        return self.input_map(x)
