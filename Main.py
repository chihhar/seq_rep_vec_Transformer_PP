import os

dir_folder=os.getcwd()

import argparse
import numpy as np
import pandas as pd

###################################

import pdb
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import Utils
from data import data_selector
import transformer.Constants as Constants

# import plot_code
from matplotlib import pyplot as plt

from transformer.Models import Transformer
from tqdm import tqdm
from functools import partial
from datetime import datetime as dt


################
### Early Stop
################
class EarlyStopping:
    def __init__(self,patience=10, verbose=False, path='c_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.checkpoint(val_loss, model)
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1
            if self.verbose:  #表示を有b効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.checkpoint(val_loss, model)
            self.counter = 0
    def checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する

################
### Train
################
def train_epoch(model, training_data, optimizer, opt):
    """ Epoch operation in training phase. """
    model.train()
    
    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_time_ae = 0
    total_num_event = 0  # number of total events
    
    for batch in tqdm(training_data, mininterval=2,dynamic_ncols=True,
                      desc='-(Training)  ', leave=False):
        mask=None
        if len(batch)==2:
            mask=batch[1]
            batch=batch[0]
        """ prepare data """
        event_time = batch.to(opt.device, non_blocking=True)
        
        train_input = event_time[:,:-1]#[B,Seqence-1]
        train_target = event_time[:,-1:]#[B,1]
        
        """ forward """
        optimizer.zero_grad()

        #model_output [B,1,M], prediction[B,1], enc_out[B,Sequence,M]
        model_output, prediction, enc_out = model(train_input,train_target)
        
        
        """ backward """
        # negative log-likelihood
        #event_ll[B,1,1], non_event_ll.shape[B,1,1]
        if mask is not(None):
                mask=mask[:,-1:].to(opt.device)
        else:
            mask=torch.ones(event_time.shape[0],1).to(opt.device)

        event_ll, non_event_ll = Utils.log_likelihood(model, model_output, train_input, train_target, enc_out)
        mask=mask.reshape(event_ll.shape)
        event_loss = -torch.sum((event_ll - non_event_ll)*mask)
        # time prediction
        
        se = Utils.time_loss_se(prediction, train_input, train_target,mask)#[]
        with torch.no_grad():
            ae = Utils.time_loss_ae(prediction, train_input, train_target,mask)#[]
        # SE is usually large, scale it to stabilize training
        scale_time_loss = opt.loss_scale
        
        loss = event_loss + se / scale_time_loss 
        loss.backward()
        
        if opt.grad_log==True:
            for name, param in model.named_parameters():
                with open(f"./param_grad/{opt.imp}/{name}.log","a") as f:
                    f.write(f'{param.grad}\n')
        """ update parameters """
        optimizer.step()
        """ note keeping """
        
        total_event_ll += float(-event_loss.item())
        total_time_se += float(se.item())
        total_time_ae += float(ae.item())
        total_num_event += int(mask.sum().item())
        del loss
        del model_output
        del enc_out
        torch.cuda.empty_cache()
        
    
    mse = total_time_se / total_num_event
    mae = total_time_ae / total_num_event
    return total_event_ll / total_num_event, mae, mse
################
### Evaluation
################
def eval_epoch(model, validation_data, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_time_ae = 0
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
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
            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, output, train_input, train_target, enc_out)
            mask=mask.reshape(event_ll.shape)
            event_loss = -torch.sum((event_ll - non_event_ll)*mask)
            # time prediction
            se = Utils.time_loss_se(prediction, train_input, train_target,mask)
            
            ae = Utils.time_loss_ae(prediction, train_input, train_target,mask)
            """ note keeping """
            total_event_ll += float(-event_loss.item())
            total_time_se += float(se.item())
            total_time_ae += float(ae.item())
            total_num_event += int(mask.sum().item())
    mse = total_time_se / total_num_event
    mae = total_time_ae / total_num_event
    return total_event_ll / total_num_event, mae, mse

################
### train-eval-plot-earlystop
################
def train(model, training_data, validation_data ,test_data,optimizer, scheduler, opt):
    """ Start training. """
    train_loss_his = []

    valid_event_losses = []  # validation log-likelihood
    valid_mae_history = [] # validation event time prediction MAE
    valid_mse_history = []  # validation event time prediction MSE
    valid_loss_his = []
    if not os.path.exists(f"checkpoint/{opt.gene}"):# 無ければ
        os.makedirs(f"checkpoint/{opt.gene}") 
    if opt.epoch==0:
        torch.save(model.state_dict(), f"checkpoint/{opt.gene}/{opt.wp}.pth") 
        epoch = 0
    
    torch.backends.cudnn.benchmark = True
    es = EarlyStopping(verbose=True,path=f"checkpoint/{opt.gene}/{opt.wp}.pth")

    for epoch_i in range(opt.epoch):
        torch.cuda.empty_cache()
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')
        start = time.time()
        
        train_event, train_mae ,train_mse= train_epoch(model, training_data, optimizer, opt)
        print('  - (Training)    Loss:{loss: 8.5f},loglikelihood: {ll: 8.5f}, '
            ' MAE: {mae: 8.5f},'
            'RMSE: {rmse: 8.5f}, '
            'elapse: {elapse:3.3f} min'
            .format(loss=-train_event+train_mae/opt.loss_scale, ll=train_event, mae=train_mae, rmse=np.sqrt(train_mse), elapse=(time.time() - start) / 60))
        train_loss_his += [-train_event+train_mae/opt.loss_scale]
        
        start = time.time()
        valid_event, valid_mae, valid_mse = eval_epoch(model, validation_data, opt)
        print('  - (Valid   )    Loss:{loss: 8.5f},loglikelihood: {ll: 8.5f}, '
            ' MAE: {mae: 8.5f},'
            ' RMSE: {rmse: 8.5f}, '
            'elapse: {elapse:3.3f} min'
            .format(loss=-valid_event+ valid_mae/opt.loss_scale, ll=valid_event, mae=valid_mae, rmse=np.sqrt(valid_mse), elapse=(time.time() - start) / 60))
        valid_loss_his +=[-valid_event+ valid_mse/opt.loss_scale]
        valid_event_losses += [valid_event]
        valid_mae_history += [valid_mae]
        valid_mse_history += [valid_mse]
        print('  - [Info] Loss: {loss:8.5f}, Maximum ll: {event: 8.5f}, Minimum MAE: {mae: 8.5f}, Minimum RMSE:{rmse: 8.5f}'
            .format(loss=max(valid_loss_his), event=max(valid_event_losses), mae=min(valid_mae_history), rmse=min(np.sqrt(valid_mse_history))))

        # logging
        with open(opt.log, 'a') as f:
            f.write("train     : "+'{epoch}, {loss: 8.5f}, {ll: 8.5f}, {mae: 8.5f}, {rmse: 8.5f}\n'
                .format(epoch=epoch,loss=-train_event+train_mae/opt.loss_scale, ll=train_event, mae=train_mae, rmse=np.sqrt(train_mse)))
            f.write("validation: "+'{epoch}, {loss: 8.5f}, {ll: 8.5f}, {mae: 8.5f}, {rmse: 8.5f}\n\n'
                .format(epoch=epoch,loss=-valid_event+ valid_mae/opt.loss_scale, ll=valid_event, mae=valid_mae, rmse=np.sqrt(valid_mse)))

        scheduler.step()
        
        es( -valid_event+valid_mse/opt.loss_scale ,model)
        if es.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
            print("Early Stopping!")
            break

    model_path=f"checkpoint/{opt.gene}/{opt.wp}.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    start = time.time()
    test_event, test_mae, test_mse = eval_epoch(model, test_data, opt)
    print('  - (testing   )    Loss:{loss: 8.5f},loglikelihood: {ll: 8.5f}, '
            ' MAE: {mae: 8.5f},'
            ' RMSE: {rmse: 8.5f}, '
            'elapse: {elapse:3.3f} min'
            .format(loss=-test_event+ test_mse/opt.loss_scale, ll=test_event, mae=test_mae, rmse=np.sqrt(test_mse), elapse=(time.time() - start) / 60))
        
    with open(opt.log, 'a') as f:
        f.write(f"test: Loss:{-test_event+ test_mse/opt.loss_scale: 8.5f},"
                f' loglikelihood: {test_event: 8.5f},'
                f' RMSE: {np.sqrt(test_mse): 8.5f},'
                f' MAE: {test_mae: 8.5f}\n\n')
    
    with open(opt.log, 'a') as f:
        f.write(f'rep values {model.rep_vector}\n')
        f.write(f'anchor values {model.anchor_vector}\n')
    #plot_code.plot_learning_curve(train_loss_his,valid_loss_his,opt)
        
def test(model, training_data, validation_data ,test_data,optimizer, scheduler, opt):
    path=opt.wp
    model_path=f"checkpoint/{opt.gene}/{opt.wp}phase3.pth"
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    start = time.time()
    test_event, test_mae, test_mse = eval_epoch(model, test_data, opt)
    print('  - (testing   )    Loss:{loss: 8.5f},loglikelihood: {ll: 8.5f}, '
                ' MAE: {mae: 8.5f},'
                ' MSE: {mse: 8.5f}, '
                ' RMSE: {rmse: 8.5f}, '
                'elapse: {elapse:3.3f} min'
                .format(loss=-test_event+ test_mse/opt.loss_scale, ll=test_event, mae=test_mae, mse=test_mse,rmse=np.round(np.sqrt(test_mse),5), elapse=(time.time() - start) / 60))
    # plot_code.phase_eventGT_prediction_plot(model, test_data,opt)
    # attention
    # seq-rep
    # tsne
    # hazard
#################
### Main
#################
def main():#python Main.py --train -gene="jisin"
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
    
    ''' set up folders'''
    check_list=[f"checkpoint/{opt.gene}",
                f"log",
                f"pickled/proposed/{opt.gene}",
                f"pickled/proposed/{opt.gene}/rep/",
                f"pickled/proposed/{opt.gene}/anc/",
                f"pickled/GT/{opt.gene}/",
                f"plot/loss_lc/{opt.imp}/",
                f"plot/event_GT/{opt.gene}/{opt.trainvec_num}_{opt.pooling_k}/"
                f"plot/ronb/",
                f"plot/t_SNE/rep/",
                f"plot/t_SNE/anc/",
                f"plot/all_eve_pred/{opt.gene}/"
                ]
    for check_txt in check_list:
        if not os.path.exists(f"{check_txt}"):
            os.makedirs(f"{check_txt}")

    # setup the log file
    if opt.train==True:
        with open(opt.log, 'w') as f:
            f.write('Loss,  Epoch,  Log-likelihood,  MAE,  MSE\n')
    else:
        print("Not training")

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
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"{params}")  
    if opt.train == True:
        with open(opt.log, 'a') as f:
            f.write(f'anchor values {model.anchor_vector}\n')
            f.write(f'rep values {model.rep_vector}\n')
        if opt.grad_log==True:
            os.makedirs('./param_grad/{opt.imp}', exist_ok = True)
            for name, param in model.named_parameters():#初期open
                with open(f"./param_grad/{opt.imp}/{name}.log", 'w') as f:
                    f.write('st')
            
    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # sum(p.numel() for  p in model.encoder.layer_stack[0].parameters())
    
    print('[Info] Number of parameters: {}'.format(num_params))
    
    """ train the model """
    if opt.train==True:
            for param in model.parameters():
                param.requires_grad = True
            model.rep_vector.requires_grad = False
            model.anchor_vector.requires_grad = False
            """ optimizer and scheduler """
            optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                opt.lr, betas=(0.9, 0.999), eps=1e-05)

            scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
            train(model, trainloader, validloader,testloader, optimizer, scheduler, opt)
            model_path=f"checkpoint/{opt.gene}/{opt.wp}.pth"
            model.load_state_dict(torch.load(model_path))
            
            torch.save(model.state_dict(), f"checkpoint/{opt.gene}/{opt.wp}phase1.pth") 
            print("phase1 fin.")
            for param in model.parameters():
                param.requires_grad = True
            model.rep_vector.requires_grad = True
            model.anchor_vector.requires_grad = False
            """ optimizer and scheduler """
            optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                opt.lr, betas=(0.9, 0.999), eps=1e-05)

            scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
            train(model, trainloader, validloader,testloader, optimizer, scheduler, opt)
            
            model.load_state_dict(torch.load(model_path))
            torch.save(model.state_dict(), f"checkpoint/{opt.gene}/{opt.wp}phase2.pth") 
            print("phase2 fin.")
            for param in model.parameters():
                param.requires_grad = True
            model.rep_vector.requires_grad = False
            model.anchor_vector.requires_grad = True
            """ optimizer and scheduler """
            optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                opt.lr, betas=(0.9, 0.999), eps=1e-05)
            scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
            train(model, trainloader, validloader, testloader, optimizer, scheduler, opt)
            model.load_state_dict(torch.load(model_path))
            torch.save(model.state_dict(), f"checkpoint/{opt.gene}/{opt.wp}phase3.pth")
            print("phase3 fin.")
        
    else:
        test(model, trainloader, validloader, testloader, optimizer, scheduler, opt)
if __name__ == '__main__':
    plt.switch_backend('agg')
    plt.figure()
    main()