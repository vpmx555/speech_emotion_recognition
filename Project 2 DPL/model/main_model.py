from model.CAT import CrossAttentionTransformer
from model.BiLSTM import BiLSTM
import torch.nn as nn
import torch
import torch.nn.functional as F


class HuMP_CAT(nn.Module):
    def __init__(self, embed_dim, num_heads, embed_dim_x1, embed_dim_x2, dropout=0.1, num_classes = 8):
        super(HuMP_CAT, self).__init__()

        # 2D Convolution cho HuBERT (kernel_size=10x15, stride=4x3)
        self.hubert_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(10, 15),
                                     stride=(4, 3))  # (B, 256, H', W')

        self.bi_lstm = BiLSTM(input_dim=40, hidden_dim=64, num_layers=2)

        # Fully Connected cho Prosody
        self.fc_relu = nn.Linear(103, 128)  # (B, 252)
        self.fc_sigmoid = nn.Linear(128, 128)  # (B, 252)

        # Cross Attention Transformer
        self.cat1 = CrossAttentionTransformer(128, 2, 512, 2, embed_dim_x1=embed_dim_x1 // 2,
                                              embed_dim_x2=embed_dim_x1 // 2)
        self.cat2 = CrossAttentionTransformer(embed_dim, num_heads, 512, 2, embed_dim_x1=embed_dim_x1,
                                              embed_dim_x2=embed_dim_x2)

        # LayerNorm + Dropout
        self.norm = nn.LayerNorm(embed_dim * 2)
        self.dropout = nn.Dropout(dropout)

        self.fc_classifier = nn.Linear(embed_dim * 2 * 2, num_classes)  # 512 * 2 -> num_classes


    def forward(self, hubert_features, mfcc_features, prosody_features, device="cpu"):
        """
        hubert_features: (B, T, 768)  →  (B, 1, T, 768)  →  (B, 256, T', W')
        mfcc_features: (B, T, 40)     →  (B, T, 256)
        prosody_features: (B, 103)    →  (B, 256)
        """

        hubert_out = hubert_features.unsqueeze(1)  # Thêm 1 chiều channel: (B, 1, T, 768)
        pad_t = 3  # Dọc (T)
        pad_w = 6  # Ngang (768) (tính tương tự)

        hubert_out = F.pad(hubert_out, (pad_w, pad_w, pad_t, pad_t), mode="constant", value=0).to(device)

        # Tiếp tục Conv2d
        hubert_out = self.hubert_conv(hubert_out).squeeze(1)  # (B, Time_step, Ft)

        B, T, Ft = hubert_out.shape

        # MFCC → BiLSTM
        mfcc_out = mfcc_features.permute(0, 2, 1)  # (B, Ft, T)
        mfcc_out = F.adaptive_avg_pool1d(mfcc_out, T).to(device)
        mfcc_out = mfcc_out.permute(0, 2, 1)  # (B, Time_step, Ft)
        mfcc_out = self.bi_lstm(mfcc_out)  # (B, T, 252)

        # Prosody → FC
        prosody_out = self.fc_relu(prosody_features)
        # prosody_out = torch.sigmoid(self.fc_sigmoid(prosody_out)).to(device)  # (B, 252)


        seq_len = mfcc_out.shape[1]  # Lấy số bước thời gian của mfcc
        prosody_out = prosody_out.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, 256)

        # Cross Attention Transformer
        cat1_out1, cat1_out2 = self.cat1(prosody_out, mfcc_out)  # (B, T, 256)
        cat1_out = torch.cat([cat1_out1, cat1_out2], dim=-1)  # (B, T, 512)

        cat2_out1, cat2_out2 = self.cat2(hubert_out, cat1_out)  # (B, T, 256)

        # Kết hợp và chuẩn hóa
        combined = torch.cat([cat2_out1, cat2_out2], dim=-1)  # (B, T, 512)
        combined = self.norm(combined)
        combined = self.dropout(combined)

        mean_pooled = combined.mean(dim=1)  # (B, 512)
        var_pooled = combined.var(dim=1)    # (B, 512)

        # 🔹 Kết hợp mean & variance
        pooled_features = torch.cat([mean_pooled, var_pooled], dim=-1)  # (B, 1024)

        # 🔹 Dự đoán lớp
        logits = self.fc_classifier(pooled_features)  # (B, num_classes)

        return logits

def check_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Khởi tạo mô hình
    model = HuMP_CAT(embed_dim=256, num_heads=4, embed_dim_x1=256, embed_dim_x2=256)
    model.to(device)

    # Tạo dữ liệu giả lập
    batch_size = 4
    seq_length = 700  # Số bước thời gian
    hubert_features = torch.randn(batch_size, seq_length, 768).to(device)  # HuBERT
    mfcc_features = torch.randn(batch_size, seq_length, 40).to(device)  # MFCC
    prosody_features = torch.randn(batch_size, 103).to(device)  # Prosody

    # Chạy forward pass
    outputs = model(hubert_features, mfcc_features, prosody_features, device)
    print("✅ Mô hình chạy thành công!")
    print(f"🔹 Output Shape: {outputs.shape}")  # (batch_size, seq_length, 512) hoặc dạng khác tuỳ thiết kế



