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


    A = (Q @ K.transpose(-1, -2) / d**0.5).softmax(-1)
    attn_out = A @ V
    return attn_out.transpose(1, 2), A


def concat_heads(input_tensor):
  return input_tensor.flatten(-2, -1)


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, fc_dim, num_heads, activation="GELU"):
        # Input shape : (B, S, H)
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.w_qkv = nn.Linear(hidden_dim, 3*hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim)
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
        attn_out = concat_heads(attn_out)
        attn_out = self.out(attn_out)
        attn_out += x

        ln_2 = self.ln_2(attn_out)
        mlp_1 = self.mlp_1(ln_2)
        x = self.activation(mlp_1)
        x = self.mlp_2(x)
        x += attn_out

        return x


class EmbeddingBlock(nn.Module):
    def __init__(self, patch_size, input_shape, hidden_dim) -> None:
        super().__init__()
        channels, height, width = input_shape
        self.patch_size = patch_size
        self.seq_len = (height * width) // patch_size**2
        self.pos_emb = nn.Embedding(num_embeddings=self.seq_len, embedding_dim=hidden_dim)
        self.proj = nn.Linear(patch_size**2 * channels, hidden_dim)
        self.projection = nn.Conv2d(channels, hidden_dim, kernel_size=patch_size, stride=patch_size)


    def forward(self, x):
        # X shape : (B, C, H, W)
        # New shape should be : (B, S, P**2 * C)
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
        "Height and width must be divisible by patch size."

        x = self.projection(x)
        x = x.view(B, C, H*W).transpose(1, 2)

        pos_emb = self.pos_emb(torch.arange(self.seq_len, device=x.device))
        pos_emb = pos_emb.unsqueeze(0)  # Shape (1, seq_len, hidden_dim) for broadcasting

        x = x + pos_emb

        return x


class ViT(nn.Module):
    def __init__(self, patch_size, input_shape, hidden_dim, fc_dim, num_heads, num_blocks, num_classes, activation="GELU"):
        super().__init__()
        self.embed = EmbeddingBlock(patch_size=patch_size, input_shape=input_shape, hidden_dim=hidden_dim)
        self.transformers = nn.Sequential(
            *(TransformerEncoder(hidden_dim, fc_dim, num_heads, activation) for _ in range(num_blocks))
        )

        self.mlp_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embed(x) # (B, seq_len, hidden_dim)
        x = self.transformers(x)
        x = x.mean(dim=1) # (B, hidden_dim)
        yhat = self.mlp_head(x) # (B, num_classes)
        return yhat
