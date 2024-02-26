# Transformer Components Implementation Adapted from Annotated Transformer:
# https://nlp.seas.harvard.edu/2018/04/03/attention.html
import pdb
import math
import torch
import torch.nn as nn 
import torch.nn.functional as F
class ScaledDotProductAttention(nn.Module):
    
    def __init__(self,temperature,attn_dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        
        
    def forward(self,query, key, value, mask=None, dropout=None):
        #query 128,1,58,64
        # mask 128,1,58,58
        d_k = query.size(-1)
        attn = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(self.temperature)
        if mask is not None:
            # small change here -- we use "1" for masked element
            attn = attn.masked_fill(mask > 0, -1e9)
        p_attn = self.dropout(F.softmax(attn, dim=-1))
        #torch.matmul(p_attn, value) 129,1,58,64
        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_input, d_model, dropout=0.1, output_linear=False):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k
        self.d_model = d_model
        self.output_linear = output_linear
        self.attention=ScaledDotProductAttention(temperature=self.d_k**0.5,attn_dropout=dropout)
        if output_linear:
            self.linears = nn.ModuleList([nn.Linear(d_input, d_model, bias=False) for _ in range(3)] + [nn.Linear(d_model, d_model), ])
        else:
            self.linears = nn.ModuleList([nn.Linear(d_input, d_model, bias=False) for _ in range(3)])
        # Q,K,Vの座標変換
        #for i in range(len(self.linears)):
            #nn.init.xavier_uniform_(self.linears[i].weight)
        self.dropout = nn.Dropout(p=dropout)
    # def count_parameters(model):
    #  return sum(p.numel() for p in model.parameters() if p.requires_grad)sum(p.numel() for p in self.model.heads[0].parameters())
    # sum(p.numel() for p in self.model.heads[0][0].parameters())
    # self.model.heads
    #### heads[0]
    # #### self_attn
    # #### linears
    # #### dropout
    # #### feed_forward
    #### heads[7]
    # (Emb)
    # (norm)
    # (softplus)
    def forward(self, q, k, v, mask):
        #query 128,58,128
        # mask 128, 1, 58, 58
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = q.size(0)#128
        len=q.size(1)
        query = self.linears[0](q).view(nbatches,len,self.n_head,self.d_k).transpose(1,2)
        key = self.linears[1](q).view(nbatches,len,self.n_head,self.d_k).transpose(1,2)
        value = self.linears[2](q).view(nbatches,len,self.n_head,self.d_k).transpose(1,2)
        # query, key, value = [
        #     l(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
        #     for l, x in zip(self.linears, (q, k, v))
        # ]
        #query128, 1, 58, 64
        #mask 128, 1, 58, 58
        
        x, attn_weight = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_head * self.d_k)
        
        
        if self.output_linear:
            return self.linears[-1](x)
        else:
            return x


class SublayerConnection(nn.Module):
    # used for residual connnection
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward=None, use_residual=False, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_residual = use_residual
        if use_residual:
            self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        self.d_model = d_model#128

    def forward(self, x, mask):
        #x B,58,128
        
        if self.use_residual:
            #self.self_attn(x, x, x, mask) B,58,64
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            
            
            #x [B,58,64]
            if self.feed_forward is not None:
                return self.sublayer[1](x, self.feed_forward)
            else:
                return x
        else:
            return self.self_attn(x, x, x, mask)

class XFMREncoder(nn.Module):
    def __init__(self, d_model, num_layers, self_attn, feed_forward, use_residual=False, dropout=0.1):
        super(XFMREncoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, self_attn, feed_forward, use_residual, dropout)
             for _ in range(num_layers)
             ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)



