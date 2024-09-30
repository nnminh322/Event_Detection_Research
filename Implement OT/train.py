import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from data import Event_Detection_Dataset
from model import EDmodel
from ot_ultis import compute_optimal_transport
import torch.nn.functional as F
import torch.optim as optim
