import torch
from torch import nn
from transformers import BertTokenizerFast, BertModel


class EDmodel(nn.Module):
    def __init__(self, model_name="bert-base-uncase", labels=[], device="cuda"):
        super().__init__()
        self.model_name = model_name
        self.labels = labels
        self.device = device

        self.bert = BertModel.from_pretrained(self.model_name)
        self.bert.to(self.device)
        self.bert.eval()

        self.label2index = {label: idx for idx, label in enumerate(self.labels)}
        self.index2label = {idx: label for label, idx in self.label2index.items()}
        self.num_labels = len(labels)

        self.hidden_size = self.bert.config.hidden_size
        self.label_embeddings = torch.nn.Embedding(self.num_labels, self.hidden_size)
        nn.init.xavier_uniform_(self.label_embeddings.weight)

        self.trigger_ffn = nn.Linear(self.hidden_size, 1)
        self.type_ffn = nn.Linear(
            self.hidden_size + self.hidden_size, 1
        )  # 2 * Hidden_size -> 1

    def forward(self, input_ids, attention_mask, token_type_ids):

        # tokenizer = BertTokenizerFast.from_pretrained(self.model_name)

        # encoding = tokenizer(
        #     sentence,
        #     is_split_into_words=True,
        #     return_tensors="pt",
        #     return_attention_mask=True,
        #     return_token_type_ids=True,
        #     padding=True,
        #     Truncation=True,
        # )

        # input_ids = encoding["input_ids"].to(self.device)
        # attention_mask = encoding["attention_mask"].to(self.device)
        # token_type_ids = encoding["token_type_tds"].to(self.device)

        with torch.no_grad:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            last_hidden_state = outputs.last_hidden_state
            e_cls = last_hidden_state[:, 0, :]  # [Batch_size, hidden_size]

        # Trigger Identification
        p_wi = torch.sigmoid(self.trigger_ffn(last_hidden_state)).squeeze(-1)

        # Type Prediction

        label_embeddings = self.label_embeddings.weight  # [Num_label, hidden_size]
        label_embeddings = label_embeddings.unsqueeze(0).repeat(input_ids.size[0], 1, 1)  # [Num_label, hidden_size] -> [Batch_size, Num_label, Hidden_size]
        e_cls = e_cls.unsqueeze(1).repeat(1, self.num_labels, 1) # [Batch_size, hidden_size] -> [Batch_size, Num_label, hidden_size]
        concat = torch.cat([label_embeddings, e_cls], dim=-1) # Concat in last size dimention

        p_ti = torch.sigmoid(self.type_ffn(concat)).squeeze(-1)

        return p_wi, p_ti, last_hidden_state, e_cls
    
    def get_label_embedding(self):
        return self.label_embeddings.weight

        # word_ids = encoding.word_ids(batch_index=0)
        # cls_embedding = last_hidden_state[0, 0, :]
        # E = [cls_embedding]

        # current_word = None
        # current_embedding = []

        # for idx, word_id in enumerate(word_ids):
        #     if word_id is None:
        #         continue
        #     if word_id != current_word:
        #         if current_embedding:
        #             averaged_embedding = torch.stack(current_embedding, dim=0).mean(
        #                 dim=0
        #             )
        #             E.append(averaged_embedding)
        #         current_word = word_id
        #         current_embedding = []
        #     current_embedding.append(last_hidden_state[0, idx, :])

        # if current_embedding:
        #     averaged_embedding = torch.stack(current_embedding, dim=0).mean(dim=0)
        #     E.append(averaged_embedding)

        # E = torch.stack(E, dim=0)
        # T = self.label_embeddings.weight

        # return E, T
