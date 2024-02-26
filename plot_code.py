from matplotlib import pyplot as plt
import torch
import os
import numpy as np
import pdb
import pickle
import transformer.Models as trm
from tqdm import tqdm
from sklearn.manifold import TSNE
import Main
import math
import Utils
import pandas as pd

import umap

dir_folder=os.getcwd()

def get_generate(data_type):
    if data_type == 'sp':
        return Main.generate_stationary_poisson()
    elif data_type == 'nsp':
        return Main.generate_nonstationary_poisson()
    elif data_type == 'sr':
        return Main.generate_stationary_renewal()
    elif data_type == 'nsr':
        return Main.generate_nonstationary_renewal()
    elif data_type == 'sc':
        return Main.generate_self_correcting()
    elif data_type == 'h1':
        return Main.generate_hawkes1()
    elif data_type == 'h2':
        return Main.generate_hawkes2()
    elif data_type == "h_fix":
        return Main.generate_hawkes_modes()
    elif data_type == 'h_fix05':
        return Main.generate_hawkes_modes05()
def rolling_matrix(x,time_step):
            x = x.flatten()
            n = x.shape[0]
            stride = x.strides[0]
            return np.lib.stride_tricks.as_strided(x, shape=(n-time_step+1, time_step), strides=(stride,stride) ).copy()

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask

###
#縦軸経過時間 横軸EventIDのplotなど
###
def Compare_event_GT_pred(model, test_data, opt):
    model.eval()
    GT_his=[]
    pred_his=[]
    
    with torch.no_grad():
        for batch in tqdm(test_data, mininterval=2,dynamic_ncols=True,
                          desc='  - (Validation) ', leave=False):
            mask=None
            if len(batch)==2:
                mask=batch[1]
                batch=batch[0]
            """ prepare data """
            event_time = batch.to(opt.device, non_blocking=True)
            test_input = event_time[:,:-1]
            test_target = event_time[:,-1:]
            if mask is not(None):
                mask=mask[:,-1].to(opt.device)
            else:
                mask=torch.ones(event_time.shape[0]).to(opt.device)
            mask=mask.reshape(test_target.shape)

            GT_his = np.append(GT_his,np.array(test_target.squeeze(-1).cpu()))
            
            _, pred, enc_out = model(test_input,test_target)
            pred = pred.reshape(test_target.shape)
            pred_his=np.append(pred_his,pred.cpu())
    print("Compare event and GT")

    plt.clf()
    plt.figure(figsize=(10,4)) 
    plt.xlabel("event ID",fontsize=18)
    plt.ylabel("elapsed time",fontsize=18)
    plt.plot(range(100),GT_his[200:300],label="ground-truth")
    plt.plot(range(100),pred_his[200:300],c="r",label="k-means anchor",linestyle="dashed")
    
    plt.legend(fontsize=18, loc='upper right')

def save_npy_synthetic(model, testloader, opt):
    
    model.eval()
    #select data:
    test_data=testloader.__iter__()
    test_datax = test_data.next()[0:30]
    pdb.set_trace()
    #prepare data:
    event_time = test_datax.to(opt.device)
    input = event_time[:,:-1]
    target = event_time[:,-1:]
    [T,score]=get_generate(opt.gene)
    test_datat=T[90000:]
    dT_test = np.ediff1d(test_datat)
    
    rt_test = torch.tensor(rolling_matrix(dT_test,opt.time_step)).to(torch.double)
    #pdb.set_trace()
    t_min=0
    t_max=target.sum()+math.pow(10,-9)
    loop_start=0
    loop_num=5000
    loop_delta = (t_max-t_min)/loop_num
    print_progress_num = 1000
    
    cumsum_tau = torch.cumsum(target,dim=0).to(opt.device)
    log_likelihood_history = []
    non_log_likelihood_history = []
    target_history = []
    calc_log_l_history = []
    with torch.no_grad():
        for t in range(loop_start,loop_num):
            if t % print_progress_num == 0:
                print(t)
            now_row_number = (target.size(0) - ( cumsum_tau > t*loop_delta+math.pow(10,-9)).sum().item())
            if now_row_number >= target.size(0):
                break
            
            now_input = input[now_row_number:now_row_number+1]
            now_target = target[now_row_number:now_row_number+1] 
            
            minus_target_value = cumsum_tau[now_row_number-1] if now_row_number >0 else 0
            variation_target = torch.tensor((t*loop_delta+math.pow(10,-9)),device=input.device)- minus_target_value
            
            variation_target = variation_target.reshape(now_target.shape)
            output, prediction, enc_out = model(now_input,variation_target)
            event_ll, non_event_ll = Utils.log_likelihood(model, output, now_input, variation_target,enc_out)            
            
            all_t = T[90029]+t*loop_delta+math.pow(10,-9)
            if opt.gene =="sp":
                log_l_t = np.log(1)
            elif opt.gene =="nsp":
                log_l_t = np.log(0.99*np.sin((2*np.pi*all_t.cpu().numpy())/20000)+1)
            elif opt.gene=="h1":
                log_l_t = np.log(0.2 + (0.8*np.exp(-(all_t.cpu().numpy() - T[T<all_t.cpu().numpy()]))).sum())
            elif opt.gene=="h_fix05":
                log_l_t = np.log(0.2 + (0.8*np.exp(-(all_t.cpu().numpy() - T[T<all_t.cpu().numpy()]))).sum())
            elif opt.gene=="h2":
                log_l_t = np.log(0.2 + (0.4*np.exp(-(all_t.cpu().numpy()-T[T<all_t.cpu().numpy()]))).sum() + (0.4*20*np.exp(-20*(all_t.cpu().numpy()-T[T<all_t.cpu().numpy()]))).sum())
            elif opt.gene=="sc":
                past_event_num = ((T<all_t.cpu().numpy()).sum())
                
                log_l_t = np.log(np.exp(all_t.cpu().numpy() - past_event_num))
            elif opt.gene=="sr":
                log_l_t = 0
            elif opt.gene=="nsr":
                log_l_t=0  
           
            calc_log_l_history = np.append(calc_log_l_history,log_l_t)
            log_likelihood_history = np.append(log_likelihood_history,event_ll.cpu().detach().numpy())
            non_log_likelihood_history =np.append(non_log_likelihood_history,non_event_ll.cpu().detach().numpy())
            target_history+=[t*loop_delta+math.pow(10,-9)]
    #np.save("npy_Matome/"+opt.method+opt.imp+opt.gene+"_calc_intensity.npy",log_likelihood_history)
    #np.save("npy_Matome/GT_intensity.npy",calc_log_l_history)
    #np.save("npy_Matome/target_history.npy",target_history)
    
    plt.clf()
    plt.figure(figsize=(10,10))
    plt.plot(target_history,calc_log_l_history,label=r"ground-truth",color="r")
    plt.scatter(cumsum_tau.cpu(),torch.zeros(cumsum_tau.shape)-2,marker='x',color="k",label="event-time")
    #THPSLOG=np.load("THP.npy")
    THP_ll=np.load(f"{dir_folder}/npy_Matome/saveTHPh164pre_l4h1_calc_intensity.npy")
    plt.plot(target_history,(THP_ll),label=r"THP",color="b",linestyle="dashdot")
    #plt.plot(target_history,THPSLOG,label=r"THP",linestyle="dashed")
    plt.plot(target_history,(log_likelihood_history),label=r"proposed method",color="g",linestyle="dotted")
    plt.ylim(-0.1,20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(r"time", fontsize=18)
    plt.ylabel(r"log-intensity", fontsize=18)
    #メモリの数値
    #×の大きさ
    #線の太さは1?
    #線の種類
    #GTを太目黒
    #THPを青　マーク付きかどっと
    #proposedを赤　破線
    #plt.title("toy data", fontsize=18)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    print("atest")
    plt.savefig("atest.pdf",bbox_inches='tight', pad_inches=0)
    plt.savefig("atest.png",bbox_inches='tight', pad_inches=0)
    with open(f"{dir_folder}/pickled/proposed/{opt.gene}/proposed_{opt.imp}{opt.gene}_intensity", 'wb') as file:
        pickle.dump(log_likelihood_history , file)
    with open(f"{dir_folder}/pickled/proposed/{opt.gene}/target_intensity", 'wb') as file:
        pickle.dump(target_history , file)
    with open(f"{dir_folder}/pickled/proposed/{opt.gene}/True_intensity", 'wb') as file:
        pickle.dump(calc_log_l_history , file)
    with open(f"{dir_folder}/pickled/proposed/{opt.gene}/eventtime_intensity", 'wb') as file:
        pickle.dump(cumsum_tau , file)
    pdb.set_trace()
    print("end")

def plot_learning_curve(train_loss_his,valid_loss_his,opt):
    plt.clf()
    plt.plot(range(len(train_loss_his)),train_loss_his,label="train_curve")
    plt.plot(range(len(valid_loss_his)),valid_loss_his,label="valid_curve")
    plt.legend()
    plt.savefig("plot/loss_lc/"+opt.wp+'.png', bbox_inches='tight', pad_inches=0)
def phase_eventGT_prediction_plot(model, test_data,opt):
    GT_history=[]
    p1_pred_history=[]
    p2_pred_history=[]
    p3_pred_history=[]
    all_event_num=0
    model.eval()
    ll_his=[]
    non_his=[]
    with torch.no_grad():
        for batch in tqdm(test_data, mininterval=2,dynamic_ncols=True,
                          desc='  - (Validation) ', leave=False):
            mask=None            
            if len(batch)==2:
                mask=batch[1]
                batch=batch[0]
            #prepare data
            event_time = batch.to(opt.device, non_blocking=True)
            train_input = event_time[:,:-1]
            test_target = event_time[:,-1:]
            
            if mask is not(None):
                mask=mask[:,-1].to(opt.device)
            else:
                mask=torch.ones(event_time.shape[0]).to(opt.device)
            mask=mask.reshape(test_target.shape)
            #forward
            output, prediction, enc_out = model(train_input,test_target)
            event_ll, non_event_ll = Utils.log_likelihood(model, output, train_input, test_target, enc_out)
            event_ll=event_ll.reshape(mask.shape)
            non_event_ll=non_event_ll.reshape(mask.shape)
            ll_his=np.append(ll_his,np.array(event_ll[(mask>0)].cpu()))
            non_his=np.append(non_his,np.array(non_event_ll[(mask>0)].cpu()))
            prediction=prediction[(mask>0)]
            test_target=test_target[(mask>0)]
            p3_pred_history = np.append(p3_pred_history,prediction.cpu())
            GT_history = np.append(GT_history,test_target.cpu()) 
            all_event_num+=mask.sum().item()
    
   
    gosa=abs(GT_history-p3_pred_history)
    
    SE=gosa**2
    print(f"RMSE:{np.sqrt(SE.mean())}")
    print(f"rstd:{np.sqrt(np.std(SE))}")
    print(f"{opt.gene}:RMSE(std):{np.round(np.sqrt(SE.mean()),decimals=3)}({np.round(np.sqrt(np.std(SE)),decimals=3)})")
    print(f"log_likelihood:{np.round((ll_his-non_his).mean(),decimals=3)}")   
    
    dir=f"/data1/nishizawa/Desktop/Transtrans/Transformer-Hawkes-Process/pickled/proposed/{opt.gene}/"
    with open(dir+opt.imp+opt.gene+'ll', 'wb') as file:
        pickle.dump(ll_his , file)
    with open(dir+opt.imp+opt.gene+'nonll', 'wb') as file:
        pickle.dump(non_his , file)
    
    dir=f"{dir_folder}/pickled/GT/{opt.gene}/"
 
    with open(f"{dir}GT", 'wb') as file:
        pickle.dump(GT_history , file)
        
    dir=f"{dir_folder}/pickled/proposed/{opt.gene}/"
    with open(f"{dir}{opt.imp}{opt.gene}_ABS_Error", 'wb') as file:
        pickle.dump(gosa , file)

    with open(f"{dir}{opt.imp}{opt.gene}_True", 'wb') as file:
        pickle.dump(GT_history , file)

    with open(f'{dir}{opt.imp}{opt.gene}_pred', 'wb') as file:
        pickle.dump(p3_pred_history , file)

    len_error=gosa.shape[0]
    print(gosa.sum()/all_event_num)
    hako=np.array([0.25,0.5,0.75])
    gosa.sort()

    print(gosa[(len_error*hako).astype(int)])

    plt.clf()
    plt.figure(figsize=(20,4))
    plt.xlabel("event iD",fontsize=18)
    plt.ylabel("elapsed time",fontsize=18)
    print(gosa.mean())
    print(gosa.sum()/all_event_num)
    plt.plot(range(GT_history[0:100].shape[0]),GT_history[0:100],label="ground-truth")
    if opt.gene=="h_fix":
        [_,_,L_TRG1]=get_generate(opt.gene)
        test_LTRG=L_TRG1[(90000+30):]
        plt.plot(test_LTRG[0:100],linewidth=0.5,label='trig')

    plt.plot(range(p3_pred_history[0:100].shape[0]),p3_pred_history[0:100],label="p3pred",linestyle="dashdot")
    plt.legend(fontsize=18, loc='upper right')
    dir=f"{dir_folder}/plot/event_GT/{opt.gene}/{opt.trainvec_num}_{opt.pooling_k}/"
    plt.savefig(dir+opt.imp+"ID_time_"+opt.wp+".pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(dir+opt.imp+"ID_time_"+opt.wp+".svg", bbox_inches='tight', pad_inches=0)
    plt.clf()
def plot_data_hist(data,opt):
    plt.clf()
    GT_his=[]
    with torch.no_grad():
        for batch in tqdm(data, mininterval=2,dynamic_ncols=True,
                            desc='  - (Validation) ', leave=False):
                if len(batch)==2:
                    mask=batch[1]
                    batch=batch[0]
                """ prepare data """
                event_time = batch.to(opt.device, non_blocking=True)
                train_input = event_time[:,:-1]
                train_target = event_time[:,-1:]
                GT_his=np.append(GT_his,train_target.cpu())
    #pdb.set_trace()#C1=GT_his.mean() 3.394000341617557 GT_his.std()
    #Call25 66.3936274509805 73.04381631286458
    plt.title(opt.gene)
    plt.hist(GT_his,bins=100)#911
    plt.savefig(f"{dir_folder}/plot/hist/data{opt.gene}.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir_folder}/plot/hist/data{opt.gene}.svg", bbox_inches='tight', pad_inches=0)

    plt.clf()

def t_SNE(model, test_data,opt):
    plt.clf()
    enc_out_his1=None
    enc_out_his2=None
    enc_out_his3=None
    anc_his1=None
    anc_his2=None
    anc_his3=None
    GT_his=[]
    pred_his=[]
    last_event=[]
    last_weight=None
    anc_all =None
    rep_all =None
    all_event_num=0
    First_loop_flag=True

    def scorer(X,Y):
        xx=np.sqrt(((X-X.mean())**2).mean())
        yy=np.sqrt(((Y-Y.mean())**2).mean())
        xy=((X-X.mean())*(Y-Y.mean())).mean()
        rxy=xy/(xx*yy)
        return rxy
    
    with torch.no_grad():
        for batch in tqdm(test_data, mininterval=2,dynamic_ncols=True,
                          desc='  - (Validation) ', leave=False):
            mask=None
            if len(batch)==2:
                mask=batch[1]
                batch=batch[0]
            """ prepare data """
            event_time = batch.to(opt.device, non_blocking=True)
            #event_time = event_time.to(torch.float)
            train_input = event_time[:,:-1]
            test_target = event_time[:,-1:]
            if mask is not(None):
                mask=mask[:,-1].to(opt.device)
            else:
                mask=torch.ones(event_time.shape[0]).to(opt.device)
            mask=mask.reshape(test_target.shape[0])
            all_event_num+=mask.sum().item()

            """ forward """
            _, pred ,enc_out= model(train_input,test_target)
            
            test_target=test_target[(mask>0)]
            GT_his=np.append(GT_his,test_target.cpu())
            last_event = np.append(last_event,train_input[:,-1:][mask>0].cpu())

            enc_out=enc_out[(mask>0)]
            
            pred=pred[(mask>0)]
            pred_his=np.append(pred_his,pred.cpu())
            anchor_batch = model.anchor_vector.repeat([enc_out.shape[0],1])
            time_hidden = model.decoder(anchor_batch,k=enc_out,v=enc_out)
            time_pred_decout_flatten = torch.flatten(time_hidden,1)
            if First_loop_flag:
                enc_out_his1=enc_out[:,-3,:].cpu()
                enc_out_his2=enc_out[:,-2,:].cpu()
                enc_out_his3=enc_out[:,-1,:].cpu()
                rep_all=torch.flatten(enc_out[:,-3:,:],1).cpu()

                anc_his1=time_hidden[:,-3,:].cpu()
                anc_his2=time_hidden[:,-2,:].cpu()
                anc_his3=time_hidden[:,-1,:].cpu()
                anc_all=time_pred_decout_flatten.cpu()

                First_loop_flag=False    
            else:
                enc_out_his1=np.vstack((enc_out_his1,enc_out[:,-3,:].cpu()))
                enc_out_his2=np.vstack((enc_out_his2,enc_out[:,-2,:].cpu()))
                enc_out_his3=np.vstack((enc_out_his3,enc_out[:,-1,:].cpu()))
                rep_all=np.vstack((rep_all,torch.flatten(enc_out[:,-3:,:],1).cpu()))
                anc_his1=np.vstack((anc_his1,time_hidden[:,-3,:].cpu()))
                anc_his2=np.vstack((anc_his2,time_hidden[:,-2,:].cpu()))
                anc_his3=np.vstack((anc_his3,time_hidden[:,-1,:].cpu()))
                anc_all=np.vstack((anc_all,time_pred_decout_flatten.cpu()))
    #rep plot/
    with open(f"{dir_folder}/pickled/proposed/{opt.gene}/rep_{opt.imp}{opt.gene}_use_umap_Dvector.pkl", 'wb') as file:
            pickle.dump(rep_all , file)
            
    with open(f"{dir_folder}/pickled/proposed/{opt.gene}/anc_{opt.imp}{opt.gene}_use_umap_Dvector.pkl", 'wb') as file:
            pickle.dump(anc_all , file)
    for where in ["rep","anc"]:
        plot_file = f"{dir_folder}/plot/t_SNE/{where}"
        
        pickle_file = f"{dir_folder}/pickled/proposed/{opt.gene}/{where}"
        m2=umap.UMAP(n_components=2, random_state=42)
        if where == "rep":
            db = m2.fit_transform(rep_all,GT_his)
        elif where == "anc":
            db = m2.fit_transform(anc_all,GT_his)
        with open(f"{pickle_file}/proposed_{opt.imp}{opt.gene}_use_umap_2Dvector.pkl", 'wb') as file:
            pickle.dump(db , file)
        
        plt.clf()
        plt.scatter(db[:,0], db[:,1], c=GT_his,cmap='gist_stern')
        plt.xlabel("dimension 1")
        plt.ylabel("dimension 2")
        #plt.title("Transformer FT-PP, c=Ground-truth")
        plt.savefig(f"{plot_file}/GT_trainUMAP{opt.gene}{opt.imp}_c{where}.pdf")
        plt.savefig(f"{plot_file}/GT_trainUMAP{opt.gene}{opt.imp}_c{where}.png")
        plt.clf()
    
        plt.scatter(db[:,0], db[:,1], c=pred_his,cmap='gist_stern')
        plt.xlabel("dimension 1")
        plt.ylabel("dimension 2")
        plt.savefig(f"{plot_file}/pred_trainUMAP{opt.gene}{opt.imp}_c{where}.pdf")
        plt.savefig(f"{plot_file}/pred_trainUMAP{opt.gene}{opt.imp}_c{where}.png")
        plt.clf()
        
        plt.scatter(db[:,0], db[:,1], c=last_event,cmap='gist_stern')
        plt.xlabel("dimension 1")
        plt.ylabel("dimension 2")
        plt.savefig(f"{plot_file}/lasteve_trainUMAP{opt.gene}{opt.imp}_c{where}.pdf")
        plt.savefig(f"{plot_file}/lasteve_trainUMAP{opt.gene}{opt.imp}_c{where}.png")
        plt.clf()
    #/rep_plot
