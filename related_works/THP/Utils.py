import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import get_non_pad_mask
import pdb
import numpy as np

def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))

def compute_event(event):
    """ Log-likelihood of events. """
    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    result = torch.log(event)
    return result

def compute_integral_unbiased(model, data, input, target):#, gap_f):
    """ Log-likelihood of non-events, using Monte Carlo integration. 
        data(B,1,M)
        input[B,seq-1]
        target[B,1]
        enc_out (B,seq-1,M)
    """
    
    #THP用
    #tauのrandom値を渡して、encoder2にtempencしてやる必要があるのでは？x
    num_samples = 500
    #random samples 
    rand_time = target.unsqueeze(2) * \
                torch.rand([*target.size(), num_samples], device=data.device)#[B,1,num_samples]
    
    #rand_time /= (target + 1)#[B,M]
    #temp_output = model.decoder(rand_time,enc_out,enc_out,None)
    #B,100,M
    if model.method=="THP":
        temp_output=data[:,-2:-1,:]
    elif model.method=="mv3":
        temp_output = torch.cat((data[:,9:10,:], data[:,19:20,:], data[:,-2:-1,:]), dim=2)
    elif model.method=="mv6":
        temp_output = torch.cat((data[:,4:5,:], data[:,9:10,:], data[:,14:15,:],
                            data[:,19:20,:], data[:,24:25,:], data[:,-2:-1,:]), dim=2)
    for linear_layer in model.layer_stack:
        temp_output = linear_layer(temp_output)
    
    temp_lambda = softplus(temp_output + model.alpha * rand_time,model.beta)#[B,1,samples]
    
    all_lambda = torch.sum(temp_lambda,dim=2)/num_samples#[B,1]
    unbiased_integral = all_lambda * target #[B,1]
    
    return unbiased_integral

def log_likelihood(model, enc_output, input, target, nojump=False):
    """ Log-likelihood of sequence. """
    if nojump==True:
        if model.method=="THP":
            temp_output = enc_output[:,-2:-1,:]#128,1,64
            #B*1*M output
            
            #B*1*1
        elif model.method=="mv3":
            # 0~28:履歴、29予測対象
            # 10番　9:10, 20番 19:20, 29番 28:29=-2:-1
            temp_output = torch.cat((enc_output[:,9:10,:], enc_output[:,19:20,:], enc_output[:,-2:-1,:]), dim=2)
            #B*3*M output
            #B*1*1
        elif model.method=="mv6":
            # 0~28:履歴、29予測対象
            # 5番 4:5 10番　9:10, 15番 14:15, 
            # 20番 19:20, 25番 24:25 29番 28:29=-2:-1
            temp_output = torch.cat((enc_output[:,4:5,:], enc_output[:,9:10,:], enc_output[:,14:15,:],
                            enc_output[:,19:20,:], enc_output[:,24:25,:], enc_output[:,-2:-1,:]), dim=2)
        #temp_output=torch.flatten(temp_output,1)
        #128,1,384->128,1,32->128,1,16->128,1,1
        for linear_layer in model.layer_stack:
            temp_output = linear_layer(temp_output)
        
        all_lambda = softplus(temp_output+model.alpha*target.unsqueeze(2),model.beta)
        
    else:
        if model.method=="THP":
            temp_output = enc_output[:,-1:,:]
            
        elif model.method=="mv3":
            # 0~28:履歴、29予測対象
            # 10番　9:10, 20番 19:20, 29番 28:29=-2:-1
            temp_output = torch.cat((enc_output[:,10:11,:], enc_output[:,20:21,:], enc_output[:,-1:,:]), dim=2)
            #B*1*M output
        elif model.method=="mv6":
            # 0~28:履歴、29予測対象
            # 5番 4:5 10番　9:10, 15番 14:15, 
            # 20番 19:20, 25番 24:25 29番 28:29=-2:-1
            temp_output = torch.cat((enc_output[:,5:6,:], enc_output[:,10:11,:], enc_output[:,15:16,:],
                            enc_output[:,20:21,:], enc_output[:,25:26,:], enc_output[:,-1:,:]), dim=2)
        for linear_layer in model.layer_stack:
                temp_output = linear_layer(temp_output)
        all_hid = temp_output
        #B*1*1
        all_lambda = softplus(all_hid,model.beta)
    
    all_lambda = torch.sum(all_lambda,dim=2)#(B,sequence,type)の名残
    
    # event log-likelihood
    event_ll = compute_event(all_lambda)#[B,1]
    event_ll = torch.sum(event_ll,dim=-1)#[B]
    #B*1*1
    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(model, enc_output, input, target)#[16,1]
    non_event_ll = torch.sum(non_event_ll, dim=-1)#[B]
    #pdb.set_trace()
    return event_ll, non_event_ll
#THP
#(Pdb) non_event_ll
# tensor([4.8527e-01, 1.0478e-01, 7.9961e-01, 2.9828e-01, 8.4048e-01, 5.4620e-01,
#         1.3550e+00, 3.7421e-01, 7.1946e-03, 3.1724e-01, 9.8662e-01, 2.5296e-01,
#         6.1654e-01, 1.7121e-02, 3.0706e+00, 2.1202e+00, 3.1580e-01, 5.7824e-02,
#         1.4442e+00, 3.5599e-01, 1.3535e+00, 7.1971e-01, 2.5651e-01, 8.6686e-01,
#         1.2899e+00, 1.4239e+00, 3.2552e-01, 5.7925e-02, 3.1782e-01, 3.1779e+00,
#         1.2075e+00, 1.1143e-02, 4.6371e-01, 4.6376e-02, 1.9904e+00, 4.7313e-02,
#         1.9153e+00, 3.7211e-02, 7.9425e-01, 7.4448e-01, 1.6241e-01, 6.6488e-01,
#         7.1768e-01, 1.4297e-02, 2.7393e-02, 2.5819e-01, 5.8611e-01, 2.2787e-01,
#         6.2586e-01, 6.9721e-01, 3.9664e-01, 1.4819e+00, 2.4395e-01, 2.2793e+00,
#         1.3206e+00, 2.8769e-01, 5.8370e-01, 1.5825e-01, 1.4333e-01, 8.0862e-04,
#         2.1303e+00, 1.5895e+00, 1.3560e+00, 2.6987e-01, 2.6880e-01, 1.0296e+00,
#         1.2295e+00, 1.0154e+00, 4.3462e-02, 8.4279e-01, 5.8858e-01, 4.6548e-01,
#         9.8087e-02, 1.4084e-01, 2.0011e+00, 1.7756e+00, 2.1627e+00, 5.0133e-01,
#         5.1068e+00, 3.5180e-01, 1.8541e+00, 1.6633e+00, 6.7991e-01, 3.4241e+00,
#         1.9456e-01, 3.7035e-01, 1.9256e-01, 1.9481e-01, 1.2927e+00, 1.5067e+00,
#         1.1384e+00, 3.0154e-01, 5.1000e-01, 5.1306e-01, 5.5424e-01, 1.9220e+00,
#         7.8686e-02, 6.6160e-01, 1.1841e-01, 3.5739e-01, 2.3282e+00, 2.2819e-01,
#         4.2981e+00, 3.1993e+00, 4.9823e-02, 1.0947e+00, 4.2478e-01, 1.5935e+00,
#         7.5339e-01, 5.9415e-01, 7.8325e-01, 1.4503e-01, 1.6475e-01, 1.7440e+00,
#         1.0937e-01, 1.9782e+00, 1.5415e+00, 1.2858e-02, 1.1889e+00, 5.1276e-01,
#         2.6342e+00, 3.0170e-01, 1.2328e+00, 5.6755e-01, 2.6645e-01, 2.0404e-02,
#         4.6693e-01, 6.4745e-01], device='cuda:0', dtype=torch.float64)
# (Pdb) non_event_ll.shape
# torch.Size([128])
# THPnon_event_ll.mean()
# tensor(0.8755, device='cuda:0', dtype=torch.float64)
# (Pdb) non_event_ll.max()
# tensor(5.1133, device='cuda:0', dtype=torch.float64)

# mv6:non_event_ll.mean()
# tensor(0.3542, device='cuda:0', dtype=torch.float64)
# non_event_ll.max()
# tensor(12.3678, device='cuda:0', dtype=torch.float64)
# (Pdb) event_ll
# tensor([1.5088, 1.0916, 0.7049, 0.7739, 0.5431, 0.5075, 0.5313, 1.2043, 1.7341,
#         1.8581, 1.8305, 1.9182, 1.2906, 1.0685, 0.7469, 1.0724, 2.0678, 2.3785,
#         2.4341, 2.4422, 2.4422, 2.0865, 1.4531, 0.5899, 0.5131, 1.1594, 1.9326,
#         2.3056, 2.4422, 2.3173, 2.4422, 2.1817, 1.3614, 1.0346, 0.6456, 0.9883,
#         0.8828, 1.9548, 1.8882, 1.9376, 2.0568, 1.5858, 1.1482, 1.3399, 1.1201,
#         0.8303, 0.7546, 0.8129, 0.6992, 0.6313, 0.7342, 0.5215, 0.7545, 1.1003,
#         1.4658, 1.9353, 2.2534, 2.4422, 2.4422, 2.1953, 1.6688, 1.0407, 0.7778,
#         0.9233, 0.8402, 0.5215, 1.1960, 1.7880, 2.2764, 2.4144, 2.4422, 2.4422,
#         2.0743, 1.2971, 0.8775, 0.8259, 0.9819, 1.9408, 1.2793, 2.1437, 2.0513,
#         2.0867, 2.2314, 1.8313, 1.9668, 1.1853, 0.9230, 0.8520, 0.6019, 1.0099,
#         0.9614, 1.6565, 2.0097, 2.2186, 2.2542, 1.6793, 1.4001, 0.9237, 0.8678,
#         0.7981, 0.8897, 1.8769, 1.2123, 0.7035, 1.8970, 1.8416, 2.4168, 2.4098,
#         2.1788, 1.4236, 0.9147, 0.8327, 0.7470, 0.6603, 1.0837, 0.5820, 0.7748,
#         2.0366, 2.0727, 1.9687, 1.5256, 2.1928, 1.7059, 1.2430, 1.0389, 1.0005,
#         0.7135, 0.6299], device='cuda:0')
# (Pdb) event_ll.shape
# torch.Size([128])
def time_loss_se(prediction, input, target,mask):
    #prediction : (B,L-1,1)
    #event_time: (B,L)
    prediction = prediction[:,-1].squeeze(-1)
    target = target.reshape(prediction.shape)
    #t_1~t_L
    """ Time prediction loss. """
    # event time gap prediction
    diff = prediction - target
    mask=mask.reshape(diff.shape)
    se = torch.sum((diff * diff)*mask)
    return se

def time_loss_ae(prediction, input, target,mask):
    prediction = prediction[:,-1].squeeze(-1)  
    target = target.reshape(prediction.shape)  
    # event time gap prediction
    diff = prediction - target
    mask=mask.reshape(diff.shape)
    ae = torch.sum(torch.abs(diff)*mask)
    return ae

def time_mean_prediction(model, output, input, target, opt):
    #output[B,1,M], input[B,seq]
    left= math.pow(10,-9)*torch.ones(target.shape,dtype=torch.float64,device=opt.device)
    #[B,1]
    right=opt.train_mean*100*torch.ones(target.shape,dtype=torch.float64,device=opt.device)
    #[B,1]
    #input = torch.cat((input,target),dim=1)#THP用
    for _ in range(0,23):
        
        #THP用
        center=(left+right)/2
        center = center.reshape(target.shape)
        output, _ = model(input,center)
        _,non_event_ll=log_likelihood(model, output, input, center)
        value= non_event_ll-np.log(2)
        value = value.reshape(target.shape)#B,1
        left = (torch.where(value<0,center,left))#.unsqueeze(1)
        right = (torch.where(value>=0, center, right))#.unsqueeze(1)
    return (left+right)/2