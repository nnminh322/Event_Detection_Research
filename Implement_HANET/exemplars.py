import torch
import torch.nn as nn
import numpy as numpy
from tqdm import tqdm
from torch.utils.data import DataLoader
from configs import parse_arguments

args = parse_arguments()


class Exemplars:
    def __init__(self) -> None:
        self.learned_nums = 0
        self.memory_size = (
            args.enum * self.learned_nums if args.fixed_enum else args.enum
        )
        self.exemplars_x = []
        self.exemplars_mask = []
        self.exemplars_span = []
        self.exemplars_y = []
        self.radius = []

    def __len__(self):
        return self.memory_size

    def get_exemplar_loaders(self):
        x = [item for t in self.exemplars_x for item in t]
        y = [item for t in self.exemplars_y for item in t]
        mask = [item for t in self.exemplars_mask for item in t]
        span = [item for t in self.exemplars_span for item in t]
        return (x, mask, y, span, self.radius)
    
    def rm_exemplars(self,exemplar_num):
        if self.exemplars_x != [] and exemplar_num < len(self.exemplars_x[0]):
            self.exemplars_x = [i[:exemplar_num] for i in self.exemplars_x]
            self.exemplars_mask = [i[:exemplar_num] for i in self.exemplars_mask]
            self.exemplars_span = [i[:exemplar_num] for i in self.exemplars_span]
            self.exemplars_y = [i[:exemplar_num] for i in self.exemplars_y]
    
    
