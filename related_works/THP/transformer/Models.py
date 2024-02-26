import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer
from matplotlib import pyplot as plt


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout,device,train_max,normalize_before):
        super().__init__()

        self.d_model = d_model
        self.device=device
        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(train_max*1.5, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
            for _ in range(n_layers)])
        self.train_max=train_max
        self.normalize_before=normalize_before
        if normalize_before:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    def temporal_enc(self, time):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def forward(self, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        if non_pad_mask is not None:
            slf_attn_mask_subseq = get_subsequent_mask(event_time)
            slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_time, seq_q=event_time)
            slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        else:
            slf_attn_mask = None

        tem_enc = self.temporal_enc(event_time)#入力の時間エンコーディング
            
        enc_output = torch.zeros(tem_enc.shape,device=self.device)
        
        # initial_S=tem_enc[0,-2:-1,:]
        # H0=tem_enc[0,:-1,:]#initial
        # for_i=0
        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        #     layer_S=enc_output[0,-2:-1,:]
        #     layer_H=enc_output[0,:-1,:]
        #     if for_i==0:
        #         lS=layer_S.unsqueeze(0)
        #         lH=layer_H.unsqueeze(0)
        #     else:
        #         lS=torch.cat((lS,layer_S.unsqueeze(0)),dim=0)
        #         lH=torch.cat((lH,layer_H.unsqueeze(0)),dim=0)
        #     for_i+=1
        
        # l_num=len(self.layer_stack)
        # ev_num=event_time.size(1)-1
        # mk=["^","x","o","*"]
        # c_name=["m","b","g","r"]
        # lww=[1.0,1.0,1.0,1.5]
        # plt.clf()
        # plt.figure(figsize=(8,5))
        # plt.ylim(-1.01,1.05)
        # plt.xticks([2,10,20,30])
        # temp_sim=torch.cosine_similarity(initial_S,H0)
        # plt.plot(range(2,ev_num+2),temp_sim[:ev_num].cpu().detach(),lw=1.0,label=r"initial",color="k",marker="D", markersize=7)
        
        # for loop_layer in range(l_num):
        #     temp_sim=torch.cosine_similarity(lS[loop_layer],lH[loop_layer])
        #     plt.plot(range(2,ev_num+2),temp_sim[:ev_num].cpu().detach(),label=f"layer{loop_layer+1}",lw=lww[loop_layer],marker=mk[loop_layer], markersize=7,color=c_name[loop_layer])
        # plt.xlabel(r"past event index",fontsize=18)
        # plt.ylabel(f"similarity",fontsize=18)
        # plt.xticks(fontsize=18)
        # plt.yticks(fontsize=18)
        # plt.legend(fontsize=18, loc='lower right')
        
        # plt.savefig("plot/ronb/h173THP_Event_normDot_histi_.pdf", bbox_inches='tight', pad_inches=0)
        # plt.savefig("plot/ronb/h173THP_Event_normDot_histi_.svg", bbox_inches='tight', pad_inches=0)
        # plt.clf()
        
        # dir="/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/pickled/THP/h_fix05/"
        # import pickle
        # with open(dir+"THP_mse_nonexph_fix05_naiseki_lS", 'wb') as file:
        #     pickle.dump(lS , file)
        # with open(dir+"THP_mse_nonexph_fix05_naiseki_lH", 'wb') as file:
        #     pickle.dump(lH , file)
        # with open(dir+"THP_mse_nonexph_fix05_naiseki_initial_S", 'wb') as file:
        #     pickle.dump(initial_S , file)
        # with open(dir+"THP_mse_nonexph_fix05_naiseki_H0", 'wb') as file:
        #     pickle.dump(H0 , file)
        # if self.normalize_before==True:
        #     enc_output = self.layer_norm(enc_output)
        return enc_output

class Linear_layers(nn.Module):
    def __init__(
        self,d_model,d_out):
        super().__init__()
        self.linear = nn.Linear(d_model,d_out)
        self.relu = nn.ReLU()
    def forward(self,x):        
        out = self.linear(x)
        out = self.relu(out)
        return out

class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim,linear_layers):
        super().__init__()
        
        #layer_stack=[Linear_layers(int(dim/(2**i))) for i in range(linear_layers-1)]
        layer_stack=[Linear_layers(dim,32),Linear_layers(32, 16)]
        #16+512+2048or 6144 or 12288
        layer_stack.append(nn.Linear(16,1))
        self.layer_stack = nn.ModuleList(layer_stack)
        
        for i in range(len(self.layer_stack)-1):
            nn.init.xavier_uniform_(self.layer_stack[i].linear.weight)
        nn.init.xavier_uniform_(self.layer_stack[len(self.layer_stack)-1].weight)
    def forward(self, data):
        for linear_layer in self.layer_stack:
            data = linear_layer(data)
        return data
# class Predictor_exp(nn.Module):
#     """ Prediction of next event type. """

#     def __init__(self, dim,linear_layers):
#         super().__init__()
        
#         layer_stack=[Linear_layers(int(dim/(2**i))) for i in range(linear_layers-1)]
#         layer_stack.append(nn.Linear(int(dim/(2**(linear_layers-1))),1))
         
#         self.layer_stack = nn.ModuleList(layer_stack)
        
#         for i in range(len(self.layer_stack)-1):
#             nn.init.xavier_uniform_(self.layer_stack[i].linear.weight)
#         nn.init.xavier_uniform_(self.layer_stack[len(self.layer_stack)-1].weight)
#     def forward(self, data):
#         for linear_layer in self.layer_stack:
#             data = linear_layer(data)
#         return np.exp(data)

class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            d_model=256, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,time_step=20,device="cuda:0",train_max=0,linear_layers=3,normalize_before=True,opt=None):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            train_max=train_max,
            normalize_before=normalize_before
        )
        if opt.method == "THP":
            # prediction of next time stamp
            self.time_predictor = Predictor(d_model,linear_layers)
            layer_stack=[Linear_layers(int(d_model/(2**(i))),int(d_model/(2**(i+1)))) for i in range(linear_layers-1)]
            layer_stack.append(nn.Linear( int(d_model/(2**(linear_layers-1))),1))
            self.layer_stack = nn.ModuleList(layer_stack)
        elif opt.method =="mv3":
            # prediction of next time stamp
            use_history_num=3
            self.time_predictor = Predictor(d_model*use_history_num,linear_layers)
            #layer_stack=[Linear_layers(int(d_model/(2**(i)))) for i in range(linear_layers-1)]
            layer_stack=[Linear_layers(d_model,32) ,Linear_layers(32,16)]
            layer_stack.append(nn.Linear( 16,1))
            self.layer_stack = nn.ModuleList(layer_stack)
        elif opt.method =="mv6":
            use_history_num=6
            self.time_predictor = Predictor(d_model*use_history_num,linear_layers)
            #layer_stack=[Linear_layers(int(d_model/(2**(i)))) for i in range(linear_layers-1)]
            layer_stack=[Linear_layers(use_history_num*d_model,32) ,Linear_layers(32,16)]
            layer_stack.append(nn.Linear(16,1))
            self.layer_stack = nn.ModuleList(layer_stack)
        

        for i in range(len(self.layer_stack)-1):
            nn.init.xavier_uniform_(self.layer_stack[i].linear.weight)
        nn.init.xavier_uniform_(self.layer_stack[len(self.layer_stack)-1].weight)

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, 1)
        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))
        
        self.method=opt.method
        # if opt.imp=="exp":
        #     self.time_predictor = Predictor_exp(d_model,linear_layers)
    def forward(self, input_time, target):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input:  event_time: batch*seq_len.
                Target:     batch*1.
        Output: enc_output: batch*(seq_len-1)*model_dim;
                time_prediction: batch*seq_len.
        """
        
        ##THP
        
        non_pad_mask = get_non_pad_mask(torch.cat((input_time,target),dim=1))#τ予測に真値が使われないようにするために必要。
        enc_output = self.encoder(torch.cat((input_time,target),dim=1), non_pad_mask=non_pad_mask)# 入力をエンコーダ部へ
        if self.method=="THP":
            time_prediction = self.time_predictor(enc_output[:,-2:-1,:])
            return enc_output, time_prediction[:,-1,:] #強度関数に必要な出力, 時間予測, エンコーダの出力
        #/THP
        elif self.method=="mv3":
            # 0~28:履歴、29予測対象
            # 10番　9:10, 20番 19:20, 29番 28:29=-2:-1
            use_enc_output = torch.cat((enc_output[:,9:10,:], enc_output[:,19:20,:], enc_output[:,-2:-1,:]), dim=2)
            #pdb.set_trace()
            use_enc_output=torch.flatten(use_enc_output,1)
            time_prediction = self.time_predictor(use_enc_output)
            return enc_output, time_prediction #強度関数に必要な出力, 時間予測, エンコーダの出力
        elif self.method=="mv6":
            # 0~28:履歴、29予測対象
            # 5番 4:5 10番　9:10, 15番 14:15, 
            # 20番 19:20, 25番 24:25 29番 28:29=-2:-1
            use_enc_output = torch.cat((enc_output[:,4:5,:], enc_output[:,9:10,:], enc_output[:,14:15,:],
                                        enc_output[:,19:20,:], enc_output[:,24:25,:], enc_output[:,-2:-1,:]), dim=2)
            use_enc_output=torch.flatten(use_enc_output,1)
            time_prediction = self.time_predictor(use_enc_output)
            return enc_output, time_prediction #強度関数に必要な出力, 時間予測, エンコーダの出力
        
        