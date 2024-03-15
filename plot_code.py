from matplotlib import pyplot as plt
import torch
import os
import numpy as np
import pdb
import pickle
import argparse
from transformer import Models
from tqdm import tqdm
from transformer.Models import Transformer
from sklearn.manifold import TSNE
import Main
import math
import Utils
import pandas as pd
from data import data_selector
import umap

dir_folder=os.getcwd()

# fig.4
def phase_eventGT_prediction_plot(model, test_data,opt):
    print(f"start fig.4 save func")
    # save result (fig.4)
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
    dir=f"{dir_folder}/pickled/proposed/{opt.gene}/"
    with open(f"{dir}{opt.imp}{opt.gene}ll", 'wb') as file:
        pickle.dump(ll_his , file)
    with open(f"{dir}{opt.imp}{opt.gene}nonll", 'wb') as file:
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

# fig.5
def rep_attention(model, validation_data, opt):
    print(f"start fig.5 save func")
    model.eval()
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,dynamic_ncols=True,
                          desc='  - (Validation) ', leave=False):
            mask=None
            
            if len(batch)==2:
                mask=batch[1]
                batch=batch[0]
            """ prepare data """
            event_time = batch.to(opt.device, non_blocking=True)
            train_input = event_time[:,:-1]
            train_target = event_time[:,-1:]
            
            if mask is not(None):
                mask=mask[:,-1].to(opt.device)
            else:
                mask=torch.ones(event_time.shape[0]).to(opt.device)
            mask=mask.reshape(train_target.shape)

            """ forward """
            output, prediction, enc_out = model(train_input,train_target)
            non_pad_mask = Models.get_non_pad_mask(torch.cat((train_input,torch.ones((train_input.shape[0],model.rep_vector.shape[1]),device=train_input.device)),dim=1))
            rep_batch = model.rep_vector.repeat([train_input.shape[0],1])
        
            enc_output = model.encoder(train_input,rep_Mat=rep_batch,non_pad_mask=non_pad_mask,plot=True)

# fig.6
def anc_attention(model, validation_data, opt):
    print(f"start fig.6 save func")
    model.eval()
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,dynamic_ncols=True,
                          desc='  - (Validation) ', leave=False):
            mask=None
            
            if len(batch)==2:
                mask=batch[1]
                batch=batch[0]
            """ prepare data """
            event_time = batch.to(opt.device, non_blocking=True)
            train_input = event_time[:,:-1]
            train_target = event_time[:,-1:]
            
            if mask is not(None):
                mask=mask[:,-1].to(opt.device)
            else:
                mask=torch.ones(event_time.shape[0]).to(opt.device)
            mask=mask.reshape(train_target.shape)

            """ forward """
            
            non_pad_mask = Models.get_non_pad_mask(torch.cat((train_input,torch.ones((train_input.shape[0],model.rep_vector.shape[1]),device=train_input.device)),dim=1))
            rep_batch = model.rep_vector.repeat([train_input.shape[0],1])
        
            enc_output = model.encoder(train_input,rep_Mat=rep_batch,non_pad_mask=non_pad_mask)
            enc_output = enc_output[:,-(rep_batch.shape[1]):,:]
            dec_output = model.decoder(train_target,k=enc_output,v=enc_output,plot=True)

# fig.7
def t_SNE(model, test_data,opt):
    # save result of T-SNE
    print(f"start fig.7 save func")
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
    #/rep_plot

# fig.8
def save_intensity_result(model, testloader, opt):
    print(f"start fig.8 save func")
    # for fig.8
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
    
    with open(f"{dir_folder}/pickled/proposed/{opt.gene}/proposed_{opt.imp}{opt.gene}_intensity", 'wb') as file:
        pickle.dump(log_likelihood_history , file)
    with open(f"{dir_folder}/pickled/proposed/{opt.gene}/target_intensity", 'wb') as file:
        pickle.dump(target_history , file)
    with open(f"{dir_folder}/pickled/proposed/{opt.gene}/True_intensity", 'wb') as file:
        pickle.dump(calc_log_l_history , file)
    with open(f"{dir_folder}/pickled/proposed/{opt.gene}/eventtime_intensity", 'wb') as file:
        pickle.dump(cumsum_tau , file)


def plot_learning_curve(train_loss_his,valid_loss_his,opt):
    plt.clf()
    plt.plot(range(len(train_loss_his)),train_loss_his,label="train_curve")
    plt.plot(range(len(valid_loss_his)),valid_loss_his,label="valid_curve")
    plt.legend()
    plt.savefig("plot/loss_lc/"+opt.wp+'.png', bbox_inches='tight', pad_inches=0)


def main():
    """ Main function. """
    parser = argparse.ArgumentParser()
    parser.add_argument("-method",type=str, default="both_scalar")
    parser.add_argument('-epoch', type=int, default=1000)
    parser.add_argument('-batch_size', type=int, default=128)#32
    parser.add_argument('-loss_scale',type=int,default=1)

    parser.add_argument('-d_model', type=int, default=64)#512
    parser.add_argument('-d_inner_hid', type=int, default=64)#1024
    parser.add_argument('-d_k', type=int, default=8)#512
    parser.add_argument('-d_v', type=int, default=8)#512

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-linear_num', type=int, default=3)
    

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)#1e-5
    parser.add_argument('-smooth', type=float, default=0.1)
    parser.add_argument('-gene', type=str, default='h1')
    parser.add_argument('-log', type=str, default='log/log.txt')
    
    parser.add_argument('-imp', type=str, default='_')
    parser.add_argument("-train_mean", type=float, default=0)
    parser.add_argument("-train_max", type=float, default=0)
    parser.add_argument("-train_min", type=float, default=0)
    parser.add_argument("-train_med", type=float, default=0)
    parser.add_argument("-train_std", type=float, default=0)
    
    parser.add_argument("-test_mean", type=float, default=0)

    parser.add_argument("-time_step", type=int, default=30)
    parser.add_argument("-trainvec_num",type=int, default=3)
    parser.add_argument("-pooling_k",type=int, default=3)
    
    parser.add_argument("--train",action="store_true")
    parser.add_argument("--pickle_F",action='store_true')
    parser.add_argument("--miman",action='store_true')

    
    parser.add_argument("--kmean",action="store_true")
    parser.add_argument("--phase",action="store_true")
    parser.add_argument("--normalize",action="store_true")
    parser.add_argument("--grad_log",action="store_true")
    parser.add_argument("-wp",type=str, default="_")
    parser.add_argument("-device_num",type=int, default=0)
    opt = parser.parse_args()
    opt.train=False
    opt.pre_attn=True
    opt.phase=True
    # default device is CUDA
    opt.device = torch.device('cuda:'+str(opt.device_num))
    opt.log = f"{opt.d_model}_{opt.d_inner_hid}_{opt.d_k}_{opt.d_v}_{opt.n_head}_{opt.gene}_{opt.method}_{opt.imp}_{opt.epoch}_{opt.time_step}_{opt.trainvec_num}_{opt.pooling_k}"

    if opt.phase==True:
        opt.log+="_phase"
    if opt.pre_attn == True:
        opt.log+="_preLN"
    else:
        opt.log+="_postLN"
    
    opt.wp = opt.log
    opt.log="log/"+opt.log+"_log.txt"
    print(f'[Info] option parameters: {opt}')
    
    train_mask_path=None
    """ setting dataloader """
    trainloader, validloader, testloader = data_selector.set_data_loader(opt)
    
    """ prepare model """
    model = Transformer(
        d_model=opt.d_model,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        trainvec_num=opt.trainvec_num,
        pooling_k=opt.pooling_k,
        time_step=opt.time_step,
        device=opt.device,
        method=opt.method,
        train_max=opt.train_max,
        train_min=opt.train_min,
        train_med=opt.train_med,
        linear_layers=opt.linear_num,
        normalize_before=opt.pre_attn,
        train=trainloader,
        opt=opt
    )
    model.to(opt.device)
    #
    path=opt.wp
    model_path=f"checkpoint/{opt.gene}/{opt.wp}phase3.pth"
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    
    # update fig.4 result
    phase_eventGT_prediction_plot(model, testloader,opt)
    # update fig.5 result
    rep_attention(model, testloader, opt)
    # update fig.6 result
    anc_attention(model, testloader, opt)
    
    # update fig.7 result
    t_SNE(model, testloader,opt)
    
    if opt.gene=="h_fix05":
        # update fig.8 result
        save_intensity_result(model, testloader, opt)

if __name__=="__main__":
    main()