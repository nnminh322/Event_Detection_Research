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
    bert_model_name = 'bert-base-uncased'
    labels = ["Purchase", "Employment", "Other"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-3
    epsilon = 0.1  # Tham số regularization cho Sinkhorn
    
    # Khởi tạo tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    
    # Dữ liệu mẫu (thay thế bằng dữ liệu thực tế)
    # Mỗi mẫu là một dict: {'words': [...], 'labels': [...], 'types': [...]}
    # 'labels' chứa chỉ số nhãn cho từng từ (-1 cho không trigger)
    # 'types' chứa nhãn loại sự kiện (0 hoặc 1) cho từng loại
    data = [
        {
            'words': ["John", "bought", "a", "new", "car", "."],
            'labels': [0, 0, -1, -1, 0, -1],  # Giả sử không từ nào là trigger
            'types': [1, 0, 0]  # 'Purchase' hiện diện
        },
        {
            'words': ["Mary", "started", "a", "new", "job", "today", "."],
            'labels': [0, 1, -1, -1, 0, -1, -1],  # 'started' là trigger cho 'Employment'
            'types': [0, 1, 0]  # 'Employment' hiện diện
        },
        # Thêm nhiều mẫu khác
    ]
    
    # Tạo dataset và dataloader
    dataset = Event_Detection_Dataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Khởi tạo mô hình
    model = EDmodel(model_name=bert_model_name, labels=labels, device=device)
    model.train()  # Đặt mô hình ở chế độ huấn luyện
    
    # Định nghĩa optimizer (chỉ cập nhật embedding của nhãn và các head)
    optimizer = optim.Adam([
        {'params': model.label_embeddings.parameters()},
        {'params': model.trigger_ffn.parameters()},
        {'params': model.type_ffn.parameters()}
    ], lr=learning_rate)
    
    # Vòng lặp huấn luyện
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)          # [batch_size, seq_len]
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            label_ids = batch['label_ids'].to(device)          # [batch_size, seq_len]
            type_label_ids = batch['type_label_ids'].to(device)      # [batch_size, num_labels]
            
            optimizer.zero_grad()
            
            # Forward pass
            p_wi, p_tj, last_hidden_state, e_cls = model(input_ids, attention_mask, token_type_ids)
            # p_wi: [batch_size, seq_len]
            # p_tj: [batch_size, num_labels]
            # last_hidden_state: [batch_size, seq_len, hidden_size]
            # e_cls: [batch_size, hidden_size]
            
            # Tính E: vector biểu diễn từ bằng cách tính trung bình các word-pieces
            # Trong trường hợp này, E là last_hidden_state vì đã tính trung bình trong data.py
            E = last_hidden_state  # [batch_size, seq_len, hidden_size]
            T = model.get_label_embeddings()  # [num_labels, hidden_size]
            
            # Tính toán ma trận chi phí C: Khoảng cách Euclidean giữa E và T
            # E: [batch_size, seq_len, hidden_size]
            # T: [num_labels, hidden_size]
            # C: [batch_size, seq_len, num_labels]
            E_exp = E.unsqueeze(2)  # [batch_size, seq_len, 1, hidden_size]
            T_exp = T.unsqueeze(0).unsqueeze(0)  # [1, 1, num_labels, hidden_size]
            C = torch.norm(E_exp - T_exp, p=2, dim=-1)  # [batch_size, seq_len, num_labels]
            
            # Tính p(x) và q(y) bằng softmax trên p_wi và p_tj
            D_W_P = F.softmax(p_wi, dim=1)  # [batch_size, seq_len]
            D_T_P = F.softmax(p_tj, dim=1)  # [batch_size, num_labels]
            
            # Tính toán ma trận căn chỉnh A bằng OT
            pi_star = compute_optimal_transport(D_W_P, D_T_P, C, epsilon=epsilon)  # [batch_size, seq_len, num_labels]
            
            # Tính L_task: Negative Log-Likelihood Loss
            # y_l = chỉ số nhãn đúng cho từng từ
            # Nếu từ không trigger, y_l = 'Other' (chỉ số cuối cùng)
            other_label = model.num_labels - 1
            y_l = torch.where(label_ids != -100, label_ids, torch.tensor(other_label, device=device))
            # y_l: [batch_size, seq_len]
            
            # Lấy pi_star tại các nhãn đúng
            pi_star_golden = pi_star.gather(2, y_l.unsqueeze(2)).squeeze(2)  # [batch_size, seq_len]
            
            # Tính L_task
            L_task = F.binary_cross_entropy(pi_star_golden, (label_ids != -100).float(), reduction='mean')
            
            # Tính pi_g: ma trận căn chỉnh vàng
            # pi_g[w, l] =1 nếu l là nhãn đúng cho w, else 0
            pi_g = F.one_hot(y_l, num_classes=model.num_labels).float()  # [batch_size, seq_len, num_labels]
            
            # Tính toán khoảng cách Wasserstein
            Dist_pi_star = (pi_star * C).sum(dim=[1,2])  # [batch_size]
            Dist_pi_g = (pi_g * C).sum(dim=[1,2])      # [batch_size]
            
            # Tính L_OT
            L_OT = torch.abs(Dist_pi_star - Dist_pi_g).mean()
            
            # Tính LT_I: Trigger Identification Loss
            # L2.1: Binary Cross Entropy giữa p_wi và mask (trigger labels)
            LT_I = F.binary_cross_entropy(p_wi, (label_ids != -100).float(), reduction='mean')
            
            # Tính LT_P: Type Prediction Loss
            # L2.2: Binary Cross Entropy giữa p_tj và type_labels
            LT_P = F.binary_cross_entropy(p_tj, type_label_ids, reduction='mean')
            
            # Tính tổng hàm mất mát
            alpha_task = 1.0
            alpha_OT = 1.0
            alpha_LT_I = 1.0
            alpha_LT_P = 1.0
            L = alpha_task * L_task + alpha_OT * L_OT + alpha_LT_I * LT_I + alpha_LT_P * LT_P
            
            # Backward pass
            L.backward()
            optimizer.step()
            
            total_loss += L.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
