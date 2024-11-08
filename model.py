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
        # Input shape : (B, S, P, P, C)
        # Reshape : (B, S, C, P**2)
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

