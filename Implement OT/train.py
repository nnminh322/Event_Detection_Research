import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from data import Event_Detection_Dataset
from model import EDmodel
from ot_utils import compute_optimal_transport
import torch.nn.functional as F
import torch.optim as optim


def main():
    bert_model_name = "bert-base-uncased"
    labels = ["pucharse", "employee", "other"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    learning_rate = 3e-8
    num_epoch = 50
    epsilon = 1e-8
    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

    data = [
        {
            "words": ["John", "buy", "a", "new", "car", "."],
            "labels": [0, 0, -1, -1, 0, -1],
            "type": [1, 0, 0],
        },
        {
            "words": ["Mary", "started", "a", "new", "job", "today", "."],
            "labels": [0, 1, -1, -1, 0, -1, -1],
            "types": [0, 1, 0],
        },
        {
            "words": [
                "The",
                "company",
                "announced",
                "a",
                "new",
                "product",
                "launch",
                "next",
                "week",
                ".",
            ],
            "labels": [0, 0, 1, -1, -1, 0, -1, -1, -1, -1],
            "types": [1, 0, 1],
        },
        {
            "words": [
                "He",
                "completed",
                "the",
                "marathon",
                "in",
                "record",
                "time",
                ".",
            ],
            "labels": [0, 1, -1, 0, -1, 0, 0, -1],
            "types": [0, 1, 1],
        },
        {
            "words": [
                "Susan",
                "won",
                "the",
                "first",
                "prize",
                "in",
                "the",
                "competition",
                ".",
            ],
            "labels": [0, 1, -1, 0, 0, -1, -1, 0, -1],
            "types": [1, 0, 1],
        },
        {
            "words": [
                "The",
                "weather",
                "forecast",
                "predicted",
                "rain",
                "for",
                "the",
                "weekend",
                ".",
            ],
            "labels": [0, 0, 0, 1, 0, -1, -1, 0, -1],
            "types": [0, 1, 0],
        },
        {
            "words": [
                "New",
                "technology",
                "has",
                "changed",
                "the",
                "way",
                "we",
                "live",
                "and",
                "work",
                ".",
            ],
            "labels": [-1, 0, 1, 1, -1, -1, -1, 0, -1, 0, -1],
            "types": [1, 0, 1],
        },
        {
            "words": [
                "The",
                "team",
                "celebrated",
                "their",
                "victory",
                "last",
                "night",
                ".",
            ],
            "labels": [0, 0, 1, 0, 0, -1, -1, -1],
            "types": [1, 1, 0],
        },
        {
            "words": [
                "Alice",
                "is",
                "preparing",
                "for",
                "her",
                "exams",
                "next",
                "month",
                ".",
            ],
            "labels": [0, 1, 1, -1, -1, 0, -1, -1, -1],
            "types": [0, 1, 0],
        },
        {
            "words": [
                "They",
                "built",
                "a",
                "new",
                "library",
                "in",
                "the",
                "city",
                "center",
                ".",
            ],
            "labels": [0, 1, -1, -1, 0, -1, -1, 0, 0, -1],
            "types": [1, 0, 1],
        },
        {
            "words": ["The", "festival", "will", "take", "place", "in", "October", "."],
            "labels": [0, 0, 1, 1, -1, -1, -1, -1],
            "types": [1, 1, 0],
        },
        {
            "words": [
                "Climate",
                "change",
                "is",
                "affecting",
                "wildlife",
                "around",
                "the",
                "world",
                ".",
            ],
            "labels": [0, 1, 1, 1, 0, -1, -1, 0, -1],
            "types": [0, 1, 1],
        },
    ]

    dataset = Event_Detection_Dataset(data, tokenizer=tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    model = EDmodel(model_name=bert_model_name, labels=labels, device=device)
    model.train()
    model.to(device)

    optimzer = optim(
        [
            {"params": model.label_embeddings.parameters()},
            {"params": model.trigger_ffn.parameters()},
            {"params": model.type_ffn.parameters()},
        ],
        lr=learning_rate,
    )

    for epoch in num_epoch:
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            label_ids = batch["label_ids"].to(device)
            
            other_label = model.num_labels - 1


if __name__ == "__main__":
    main()
