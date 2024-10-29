import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from data import Event_Detection_Dataset
from model import EDmodel
from ot_utils import compute_optimal_transport
import torch.nn.functional as F
import torch.optim as optim


def main():
    bert_model_name = "bert-base-uncase"
    labels = []  # ...

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-3
    epsilon = 0.1

    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

    data = []  # ...

    dataset = Event_Detection_Dataset(data=data, tokenizer=tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    model = EDmodel(model_name=bert_model_name, labels=labels, device=device)
    model.train()

    optimizer = optim.adam(
        [
            {"params": model.label_embeddings.parameters()},
            {"params": model.trigger_ffn.parameters()},
            {"params": model.type_ffn.parameters()},
        ],
        learning_rate=learning_rate,
    )

    for epoch in range(num_epochs):
        total_loss = 0.0
        #####