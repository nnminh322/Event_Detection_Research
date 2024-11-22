import torch
from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F
from configs import parse_arguments
from torch.nn.utils.rnn import unpad_sequence
from random import shuffle
from utils.Optimal_Transport import OptimalTransportLayer

args = parse_arguments()
device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")  # type: ignore


class BertED(nn.Module):
    def __init__(self, class_num=args.class_num + 1, input_map=False):
        super().__init__()
        self.backbone = BertModel.from_pretrained(args.backbone)
        if not args.no_freeze_bert:
            print("Freeze bert parameters")
            for _, param in list(self.backbone.named_parameters()):
                param.requires_grad = False
        else:
            print("Update bert parameters")
        self.is_input_mapping = input_map
        self.input_dim = self.backbone.config.hidden_size

        self.label_embeddings = torch.rand(
            [class_num, self.backbone.config.hidden_size], requires_grad=True
        ).to(device)
        self.trigger_ffn = nn.Linear(2 * self.backbone.config.hidden_size, 1)
        self.type_ffn = nn.Linear(
            2 * self.backbone.config.hidden_size, 1
        )  # 2 * Hidden_size -> 1

        self.OT_layer = OptimalTransportLayer()

    def forward(self, x, masks, span=None, aug=None):
        # x = self.backbone(x) #TODO: test use
        return_dict = {}
        backbone_output = self.backbone(x, attention_mask=masks)
        x, pooled_feat = backbone_output[0], backbone_output[1]
        context_feature = x.view(-1, x.shape[-1])
        return_dict["reps"] = x[:, 0, :].clone()
        if span != None:
            outputs, trig_feature = [], []
            for i in range(len(span)):
                # if self.is_input_mapping:
                #     x_cdt = torch.stack(
                #         [
                #             torch.index_select(x[i], 0, span[i][:, j])
                #             for j in range(span[i].size(-1))
                #         ]
                #     )
                #     x_cdt = x_cdt.permute(1, 0, 2)
                #     x_cdt = x_cdt.contiguous().view(x_cdt.size(0), x_cdt.size(-1) * 2)
                #     opt = self.input_map(x_cdt)
                # else:
                opt = torch.cat(
                    [
                        torch.index_select(x[i], 0, span[i][:, 0]),
                        torch.index_select(x[i], 0, span[i][:, 1]),
                    ],
                    dim=-1,
                )

                trig_feature.append(opt)

            trig_feature = torch.cat(trig_feature)
        p_wi = torch.sigmoid(self.trigger_ffn(trig_feature))
        D_W_P = torch.softmax(p_wi)

        outputs = self.fc(trig_feature)
        return_dict["outputs"] = outputs
        return_dict["context_feat"] = context_feature
        return_dict["trig_feat"] = trig_feature
        if aug is not None:
            feature_aug = trig_feature + torch.randn_like(trig_feature) * aug
            outputs_aug = self.fc(feature_aug)
            return_dict["feature_aug"] = feature_aug
            return_dict["outputs_aug"] = outputs_aug
        return return_dict

    def forward_backbone(self, x, masks):
        x = self.backbone(x, attention_mask=masks)
        x = x.last_hidden_state
        return x

    def forward_input_map(self, x):
        return self.input_map(x)
