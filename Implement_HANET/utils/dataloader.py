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
    
    
