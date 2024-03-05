from copy import deepcopy as c
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


device = th.device('cuda:0')


class PositionalEncoding(nn.Module):
    # 这里的d_model指的是对于encoder和decoder的H*df，即多头注意力下得到的最终的特征维度空间
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.register_buffer('pos_table', self.get_table(max_len, d_model))
        self.dropout = nn.Dropout(dropout)

    def get_table(self, max_len, d_model):
        table = th.zeros(max_len, d_model).to(device)
        pos = th.arange(0., max_len).unsqueeze(1)
        # 这里是为了防止pow计算太慢对原公式先取log再用e为底取回来，这么简单的问题我在这里居然卡了好久
        div_term = th.exp(th.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        table[:, 0::2] = th.sin(pos * div_term)
        table[:, 1::2] = th.cos(pos * div_term)
        return table.unsqueeze(0)

    def forward(self, x):
        # x:[m, k, d]
        x = x + self.pos_table.clone().detach()
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # 这里转置直接用倒数是因为前面各种乱七八糟的维度太多考虑太复杂，不如干脆取最后两维
        attention = th.matmul(q / self.temperature, k.transpose(-1, -2))
        if mask is not None:
            attention = attention.masked_fill(mask == 1, -1e9)

        attention = self.dropout(F.softmax(attention, dim=-1))
        out = th.matmul(attention, v)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, d_f, local_context_length, relation, dropout=0.1):
        super().__init__()
        self.relation = relation
        self.d_f = d_f
        self.h = h
        self.l = local_context_length
        self.dropout = nn.Dropout(dropout)
        self.linear_Q = nn.Linear(d_model, d_f * h, bias=False)
        self.linear_K = nn.Linear(d_model, d_f * h, bias=False)
        self.linear_V = nn.Linear(d_model, d_f * h, bias=False)
        self.fc = nn.Linear(d_f * h, d_model, bias=False)
        self.attention = ScaledDotProductAttention(d_f, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):

        residual = query
        m = query.size(0)
        k = query.size(1)
        klmask = np.ones([k, k+self.l-1])
        for i in range(k):
            for j in range(i, i+self.l):
                klmask[i, j] = 0
        klmask = th.from_numpy(klmask).byte().to(device)
        query = self.linear_Q(query)
        local_query_q = query.view(m, self.h, k, self.d_f)
        local_query_k = th.ones(m, self.h, self.l-1, self.d_f).to(device)
        local_query_v = th.ones(m, self.h, self.l-1, self.d_f).to(device)
        local_query_k = th.cat([local_query_k, local_query_q], 2)
        local_query_v = th.cat([local_query_v, local_query_q], 2)
        tq = th.matmul(local_query_q, local_query_k.transpose(-1, -2)) / math.sqrt(self.d_f)
        tq = tq.masked_fill(klmask == 1, 0.)
        Q = th.matmul(tq, local_query_v)
        key = self.linear_K(key)
        local_key_k = key.view(m, self.h, k, self.d_f)
        local_key_q = th.ones(m, self.h, self.l-1, self.d_f).to(device)
        local_key_v = th.ones(m, self.h, self.l-1, self.d_f).to(device)
        local_key_q = th.cat([local_key_q, local_key_k], 2)
        local_key_v = th.cat([local_key_v, local_key_k], 2)
        tk = th.matmul(local_key_k, local_key_q.transpose(-1, -2)) / math.sqrt(self.d_f)
        tk = tk.masked_fill(klmask == 1, 0.)
        K = th.matmul(tk, local_key_v)
        value = self.linear_V(value)
        V = value.view(m, self.h, k, self.d_f)
        out = self.attention(Q, K, V, mask)
        if self.relation is True:
            out = out.view(self.h, k, m, self.d_f)
            out1 = out
            out2 = out
            out3 = out
            out = self.attention(out1, out2, out3)

        out = out.view(m, k, self.h * self.d_f)
        out = self.fc(out)
        residual = residual + self.fc(out.view(m, k, self.h * self.d_f))
        out = self.layer_norm(residual)
        return out.to(device)


class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.cov = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(inplace=False),
            nn.Linear(d_hid, d_in),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.cov(x)


class EncoderLayer(nn.Module):
    def __init__(self, h, d_model, d_f, local_context_length, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(h, d_model, d_f, local_context_length, True, dropout)
        self.ff = FeedForward(d_model, d_f, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        enc_out = self.attn(x, x, x)
        residual = enc_out
        enc_out = self.ff(enc_out)
        enc_out = enc_out + residual
        return self.layer_norm(enc_out).to(device)


class Encoder(nn.Module):
    def __init__(self, k, n_layers, n_head, d_f, d_model, local_context_length, dropout=0.1):
        super().__init__()
        self.position_enc = PositionalEncoding(d_model, dropout, k)
        self.dropout = nn.Dropout(dropout)
        layers = []
        for _ in range(n_layers):
            layer = EncoderLayer(n_head, d_model, d_f, local_context_length, dropout)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq):
        enc_output = self.dropout(self.position_enc(src_seq))
        enc_output = self.layer_norm(enc_output)
        for enc_layer in self.layers:
            enc_output = enc_layer(enc_output)

        return enc_output


class DecoderLayer(nn.Module):
    def __init__(self, h, d_model, d_f, local_context_length, dropout=0.1):
        super().__init__()
        self.attn1 = MultiHeadAttention(h, d_model, d_f, local_context_length, True, dropout)
        self.attn2 = MultiHeadAttention(h, d_model, d_f, local_context_length, False, dropout)
        self.ff = FeedForward(d_model, d_f, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, mask):
        dec_out = self.attn1(x, x, x, mask)
        residual = dec_out
        dec_out = self.attn2(enc_out, enc_out, dec_out)
        dec_out = self.ff(dec_out)
        dec_out = dec_out + residual
        return self.layer_norm(dec_out)


class Decoder(nn.Module):
    def __init__(
            self, k, n_layers, n_head, d_f,
            d_model, local_context_length, dropout=0.1):
        super().__init__()
        self.position_enc = PositionalEncoding(d_model, dropout, k)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(n_head, d_model, d_f, local_context_length, dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, enc_out, mask):
        dec_output = self.dropout(self.position_enc(src_seq))
        dec_output = self.layer_norm(dec_output)
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, enc_out, mask)

        return dec_output


def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), 1, seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = th.from_numpy(subsequence_mask).byte()
    return subsequence_mask.to(device)


class RAT(nn.Module):
    def __init__(self, k, n_layers, n_head, d_f, d_model, local_context_length, dropout=0.01):
        super().__init__()
        self.encoder = Encoder(k, n_layers, n_head, d_f, d_model, local_context_length, dropout)
        self.decoder = Decoder(local_context_length, n_layers, n_head, d_f, d_model, 1, dropout)
        self.linear1 = nn.Linear(k, local_context_length, bias=False)
        self.linear2 = nn.Linear(local_context_length, 1)
        self.linear4 = nn.Linear(d_model, 1)
        self.linear5 = nn.Linear(d_model, 1)

    def forward(self, enc_inputs, dec_inputs, last_ps):
        mask = get_attn_subsequence_mask(dec_inputs)
        enc_outputs = self.encoder(enc_inputs)
        enc_outputs = self.linear1(enc_outputs.transpose(-1, -2)).transpose(-1, -2)
        dec_outputs = self.decoder(dec_inputs, enc_outputs, mask)
        out = self.linear2(dec_outputs.transpose(-1, -2))
        out = F.softmax(out, dim=0).squeeze(2)
        out = out + last_ps.unsqueeze(1).detach()
        out1 = self.linear4(out).squeeze(-1)
        out2 = self.linear5(out).squeeze(-1)
        print(out1)
        out1 = F.softmax(out1)
        out2 = F.softmax(out2)
        return out1 * 2 - out2
