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
    
