from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from configs import parse_arguments
from utils.tools import collect_from_json

args = parse_arguments()


class MAVEN_Dataset(Dataset):
    def __init__(self, tokens, labels, masks, spans) -> None:
        super(Dataset).__init__()
        self.tokens = tokens
        self.labels = labels
        self.masks = masks
        self.spans = spans

    def __getitem__(self, index):
        return [
            self.tokens[index],
            self.labels[index],
            self.masks[index],
            self.spans[index],
        ]

    def __len__(self):
        return len(self.labels)

    def extend(self, tokens, labels, masks, spans):
        self.tokens.extend(tokens)
        self.labels.extend(labels)
        self.masks.extend(masks)
        self.spans.extend(spans)



def collect_dataset(dataset_name, root, split, label2idx, stage_id, labels):
    if split == "train":
        data = [
            instance
            for t in collect_from_json(
                dataset_name=dataset_name, root=root, split=split
            )[stage_id]
            for instance in t
        ]
    else:
        data = collect_from_json(root=root, dataset_name=dataset_name, split=split)
    data_tokens, data_labels, data_masks, data_spans = [], [], [], []
    for dt in tqdm(data):
        if "mention_id" in dt.keys():
            dt.pop("mention_id")
        if "sentence_id" in dt.keys():
            dt.pop("sentence_id")
        add_label, add_span, new_t = [], [], {}

        for i in range(len(dt['label'])):
            if dt['label'][i] in labels or dt['label'][i] == 0:
                add_label.append(dt['label'][i])
                add_span.append(dt['span'][i])
            



