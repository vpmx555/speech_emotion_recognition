import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(BiLSTM, self).__init__()

        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        self.layer_norm = nn.LayerNorm(2 * hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        lstm_out, _ = self.bilstm(x)  # (batch_size, seq_length, 2*hidden_dim)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        return lstm_out  # Trả về đặc trưng đã trích xuất


# Test model
if __name__ == "__main__":
    batch_size = 4
    seq_length = 10
    input_dim = 64
    hidden_dim = 128

    model = BiLSTM(input_dim, hidden_dim)
    x = torch.randn(batch_size, seq_length, input_dim)  # (batch, seq, feature_dim)
    features = model(x)

    print("Input shape:", x.shape)
    print("Feature shape:", features.shape)  # (batch_size, seq_length, 2 * hidden_dim)
