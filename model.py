import torch
import torch.nn as nn

def split_into_heads(Q, K, V, num_heads):
    Q = Q.reshape(Q.shape[0], Q.shape[1], num_heads, -1)
    K = K.reshape(K.shape[0], K.shape[1], num_heads, -1)
    V = V.reshape(V.shape[0], V.shape[1], num_heads, -1)
    return Q, K, V


def head_level_self_attention(Q, K, V):
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)
    d = Q.shape[-1]


    A = (Q @ K.transpose(-1, -2) / d**0.5).softmax(-2)
    attn_out = A @ V
    return attn_out.transpose(1, 2), A


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, patch_dim, fc_dim, num_heads, activation="relu"):
        # Input shape : (B, S, H)
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.w_qkv = nn.Linear(hidden_dim, 3*hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp_1 = nn.Linear(hidden_dim, fc_dim)
        self.mlp_2 = nn.Linear(fc_dim, hidden_dim)
        self.activation = getattr(nn.functional, activation)
        self.num_heads = num_heads

    def forward(self, x):
        ln_1 = self.ln_1(x)
        x_qkv = self.w_qkv(ln_1)
        Q, K, V = x_qkv.chunk(3, -1)
        Q, K, V = split_into_heads(Q, K, V, num_heads=self.num_heads)
        attn_out, _ = head_level_self_attention(Q, K, V)
        attn_out += x

        ln_2 = self.ln_2(attn_out)
        mlp_1 = self.mlp_1(ln_2)
        x = self.activation(mlp_1)
        x = self.mlp_2(x)
        x += attn_out

        return x


class EmbeddingBlock(nn.Module):
    def __init__(self, patch_size, height, width, hidden_dim, channels) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.seq_len = (height * width) // patch_size**2
        self.pos_emb = nn.Embedding(num_embeddings=self.seq_len, embedding_dim=hidden_dim)
        self.proj = nn.Linear(patch_size**2 * channels, hidden_dim)

    def forward(self, x):
        # X shape : (B, H, W, C)
        # New shape should be : (B, S, P**2 * C)
        batch_size, h, w, c = x.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0, \
        "Height and width must be divisible by patch size."

        # Reshape to (B, seq_len, patch_size**2 * C)
        x = x.unfold(1, self.patch_size, self.patch_size) \
             .unfold(2, self.patch_size, self.patch_size) \
             .reshape(batch_size, -1, self.patch_size**2 * c)

        x = self.proj(x)
        pos_emb = self.pos_emb(torch.arange(self.seq_len))
        pos_emb = pos_emb.unsqueeze(0)  # Shape (1, seq_len, hidden_dim) for broadcasting

        x = x + pos_emb

        return x


# TODO : Model block
# TODO : MNIST train
