import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = self.dropout(attn_output)
        return self.norm(attn_output + query)


class CrossAttentionTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, embed_dim_x1, embed_dim_x2, dropout=0.1):
        super(CrossAttentionTransformer, self).__init__()
        self.layers = nn.ModuleList([
            CrossAttention(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.proj_x1 = nn.Linear(embed_dim_x1, embed_dim)  # Chuyển x1 về cùng embed_dim
        self.proj_x2 = nn.Linear(embed_dim_x2, embed_dim)  # Chuyển x2 về cùng embed_dim

    def forward(self, x1, x2):
        x1 = self.proj_x1(x1)
        x2 = self.proj_x2(x2)
        for layer in self.layers:

            x1 = layer(x1, x2, x2)
            x2 = layer(x2, x1, x1)
        return self.norm(x1), self.norm(x2)


# Test model
if __name__ == "__main__":
    embed_dim = 56
    num_heads = 8
    ff_dim = 256
    num_layers = 2
    seq_length = 10
    batch_size = 4

    model = CrossAttentionTransformer(embed_dim, num_heads, ff_dim, num_layers, 64, 64, dropout = 0.1)
    x1 = torch.randn(seq_length, batch_size, 64)
    x2 = torch.randn(seq_length, batch_size, 64)
    print(x1.shape, x2.shape)
    out1, out2 = model(x1, x2)
    print(out1.shape, out2.shape)
