# train.py

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from data import Event_Detection_Dataset
from model import EDmodel
from ot_utils import compute_optimal_transport
import torch.nn.functional as F
import torch.optim as optim


def main():
    # Cấu hình
    bert_model_name = "bert-base-uncased"
    labels = ["Purchase", "Employment", "Other"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    num_epochs = 150
    learning_rate = 3e-6
    epsilon = 1e-8  # Tham số regularization cho Sinkhorn

    # Khởi tạo tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

    # Dữ liệu mẫu (thay thế bằng dữ liệu thực tế)
    data = [
        {
            "words": ["John", "bought", "a", "new", "car", "."],
            "labels": [0, 0, -1, -1, 0, -1],
            "types": [1, 0, 0],
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

    # Tạo dataset và dataloader
    dataset = Event_Detection_Dataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Khởi tạo mô hình
    model = EDmodel(model_name=bert_model_name, labels=labels, device=device)
    model.train()
    model.to(device)

    # Định nghĩa optimizer
    optimizer = optim.Adam(
        [
            {"params": model.label_embeddings.parameters()},
            {"params": model.trigger_ffn.parameters()},
            {"params": model.type_ffn.parameters()},
        ],
        lr=learning_rate,
    )

    # Vòng lặp huấn luyện
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            label_ids = batch["label_ids"].to(device)

            # Thay thế tất cả các giá trị âm thành `other_label`
            other_label = model.num_labels - 1
            y_l = torch.where(
                label_ids < 0, torch.tensor(other_label, device=device), label_ids
            )

            # In kiểm tra các giá trị trong y_l
            print(f"y_l unique values: {torch.unique(y_l)}")
            if not torch.all((y_l >= 0) & (y_l < model.num_labels)):
                print("Warning: Found invalid label indices in y_l after replacement")
                y_l = torch.clamp(y_l, 0, model.num_labels - 1)

            type_label_ids = batch["type_label_ids"].to(device)
            optimizer.zero_grad()

            # Forward pass
            p_wi, p_tj, last_hidden_state, e_cls = model(
                input_ids, attention_mask, token_type_ids
            )

            E = last_hidden_state
            T = model.get_label_embedding()
            E_exp = E.unsqueeze(2)
            T_exp = T.unsqueeze(0).unsqueeze(0)
            C = torch.norm(E_exp - T_exp, p=2, dim=-1)

            D_W_P = F.softmax(p_wi, dim=1)
            D_T_P = F.softmax(p_tj, dim=1)
            pi_star = compute_optimal_transport(D_W_P, D_T_P, C, epsilon=epsilon)

            # Tính L_task: Negative Log-Likelihood Loss
            pi_star_golden = pi_star.gather(2, y_l.unsqueeze(2)).squeeze(2)
            L_task = F.binary_cross_entropy(
                pi_star_golden, (label_ids >= 0).float(), reduction="mean"
            )

            pi_g = F.one_hot(y_l, num_classes=model.num_labels).float()
            Dist_pi_star = (pi_star * C).sum(dim=[1, 2])
            Dist_pi_g = (pi_g * C).sum(dim=[1, 2])
            L_OT = torch.abs(Dist_pi_star - Dist_pi_g).mean()

            LT_I = F.binary_cross_entropy(
                p_wi, (label_ids >= 0).float(), reduction="mean"
            )
            LT_P = F.binary_cross_entropy(p_tj, type_label_ids, reduction="mean")

            alpha_task = 1.0
            alpha_OT = 1.0
            alpha_LT_I = 1.0
            alpha_LT_P = 1.0
            L = (
                alpha_task * L_task
                + alpha_OT * L_OT
                + alpha_LT_I * LT_I
                + alpha_LT_P * LT_P
            )

            # Backward pass
            L.backward()
            optimizer.step()

            total_loss += L.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
