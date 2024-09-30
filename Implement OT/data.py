import torch

# from transformers import BertTokenizerFast
from torch.utils.data import Dataset


class Event_Detection_Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data(index)
        words = sample["words"]
        labels = sample["labels"]
        types = sample["types"]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True,
            padding="max_length",
            max_length=self.max_length,
            Truncation=True,
        )

        word_ids = encoding.word_ids(batch_index=0)

        label_ids = []
        for word_id in word_ids:
            if word_id is None or labels[word_id] == -1:
                label_ids.append(-1)
            else:
                label_ids.append(labels[word_id])

        type_label_ids = torch.tensor(types, dtype=torch.float)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "type_token_ids": encoding["token_type_ids"].squeeze(),
            "label_ids": label_ids,
            "type_label_ids": type_label_ids,
        }
