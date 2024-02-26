# abstract class for both trainer and tester
# Author: Chenghao Yang
from model.xfmr_nhp_fast import XFMRNHPFast
from data.NHPDataset import NHPDataset, createDataLoader
from esm.thinning import EventSampler
import os
import pickle
from tqdm import tqdm
import torch
import numpy as np
import pdb
from torchsummary import summary
class Manager:
    def __init__(self, args):
        self.args = args
        self.max_tau=0
        [self.train_loader, self.dev_loader, self.test_loader] \
            = self.get_dataloader(args)
        self.model = XFMRNHPFast(dataset=self.train_loader.dataset, d_model=args.ModelDim, n_layers=args.Layer, n_head=args.NumHead, dropout=args.Dropout,
                                 d_time=args.TimeEmbeddingDim,max_tau=self.max_tau).cuda()
        params = 0
        for p in self.model.parameters():
            if p.requires_grad:
                params += p.numel()
        #pdb.set_trace()
        print(f"parameters:{params}")

        #print(list(self.model.parameters()))
        #[p.numel for p in self.model.parameters()]
        #from torchsummary import summary
    def get_dataloader(self, args):
        loaders = []
        splits = ["train", 'dev', 'test']
        event_types = None
        token_types = 0
        
        for _split in splits:
            with open(os.path.join(args.PathDomain, f"{_split}.pkl"), "rb") as f_in:
                # latin-1 for GaTech data
                try:
                    _data = pickle.load(f_in, encoding='latin-1')
                    _data=_data
                except:
                    _data = pickle.load(f_in)
                    _data=_data
                
                # if event_types is None:
                #     event_types = _data["dim_process"]
                # else:
                #     assert _data["dim_process"] == event_types, "inconsistent dim_process in different splits?"
                event_types = len(_data)+len(_data[0])-1
                dataset = NHPDataset(_data, event_types, concurrent=False, add_bos=False, add_eos=False)
                if self.max_tau<max(max(_data)):
                    self.max_tau=max(max(_data))
                #pdb.set_trace()
                #assert dataset.event_num <= event_types, f"{_split}.pkl has more event types than specified in dim_process!"
                # token_types = max(token_types, dataset.num_types)
                loaders.append(createDataLoader(dataset, batch_size=args.BatchSize,shuffle=(_split=="train")))
        #assert token_types > event_types, f"at least we should include [PAD]! token: {token_types}, event: {event_types}"
        return loaders

    def run_one_iteration(self, model:XFMRNHPFast, dataLoader, mode, optimizer=None):
        assert mode in {"train", "eval"}
        if mode == "eval":
            model = model.eval()
        else:
            assert optimizer is not None
        total_log_like = 0
        total_acc = 0
        total_event_ll, total_non_event_ll = 0, 0
        num_tokens = 0
        pad_idx = self.train_loader.dataset.pad_index
        num_events = 0
        all_logs = []
        all_logs_token = []
        all_type_ll_token = []
        all_time_ll_token = []
        for batch in tqdm(dataLoader, mininterval=2,dynamic_ncols=True, desc=f'   - ({mode}) -    ', leave=False):
            new_batch = [x.cuda() for x in batch]
            #pdb.set_trace()
            #time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask = new_batch
            time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask = new_batch
            # summary(self.model,[(32,29),(32,29),(32,29),(32,29,29),(32,29)])
            # [(32,29),(32,29),(32,29),(32,29,29),(32,29,1)]
            # in new_batch
            event_ll, non_event_ll, enc_inten = model.compute_loglik(new_batch)
            if hasattr(self.args, "IgnoreFirst"):
                if self.args.IgnoreFirst:
                    non_event_ll[:, 0] *= 0
            _batch_loss = event_ll.sum(dim=-1) - non_event_ll.sum(dim=-1)
            _loss = -torch.sum(_batch_loss)
            if mode == "train":
                _loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            total_log_like += -_loss.item()
            total_event_ll += event_ll.sum().item()
            total_non_event_ll += non_event_ll.sum().item()
            type_lls = event_ll - torch.log(enc_inten.sum(dim=-1) + model.eps)
            time_lls = event_ll - non_event_ll - type_lls
            
            if model.add_bos:
                total_acc += ((torch.argmax(enc_inten, dim=-1) == event_seq[:, 1:]) * batch_non_pad_mask[:, 1:]).sum()
                num_tokens += event_seq[:, 1:].ne(pad_idx).sum().item()
                num_events += (event_seq[:, 1:] < pad_idx).sum().item()
                all_logs_token.extend([(x, 1.0) for x in (event_ll - non_event_ll)[batch_non_pad_mask[:, 1:]].tolist()])
                all_type_ll_token.extend([(x, 1.0) for x in type_lls[batch_non_pad_mask[:, 1:]].tolist()])
                all_time_ll_token.extend([(x, 1.0) for x in time_lls[batch_non_pad_mask[:, 1:]].tolist()])
            else:
                batch_non_pad_mask=batch_non_pad_mask[:,-1].to(torch.bool)
                total_acc += ((torch.argmax(enc_inten, dim=-1) == event_seq[:,-1]) * batch_non_pad_mask).sum()
                num_tokens += event_seq.shape[0]
                num_events += event_seq.shape[0]
                all_logs_token.extend([(x, 1.0) for x in (event_ll - non_event_ll)[batch_non_pad_mask].tolist()])
                all_type_ll_token.extend([(x, 1.0) for x in type_lls[batch_non_pad_mask].tolist()])
                all_time_ll_token.extend([(x, 1.0) for x in time_lls[batch_non_pad_mask].tolist()])
            all_logs.extend([(_batch_loss.tolist(), event_seq[:,-1].ne(pad_idx).sum(dim=-1).tolist())])
        return total_log_like, total_acc / num_tokens, (total_event_ll, total_non_event_ll), \
               num_tokens, num_events, all_logs, all_logs_token, \
               all_type_ll_token, all_time_ll_token


    def create_thinningsampler(self, num_sample, num_exp):
        self.thinning_sampler = EventSampler(num_sample, num_exp)


    def run_prediction(self, model:XFMRNHPFast, dataLoader):
        self.thinning_sampler.cuda()
        results = []
        seq_id = 0
        verbose = self.args.Verbose
        thinning_sampler = self.thinning_sampler
        se_his=[]
        pred_his=[]
        GT_his=[]
        for _batch in tqdm(dataLoader, desc=f"   (Pred)    ", leave=False,dynamic_ncols=True, mininterval=2):
            # time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask = _batch
            time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask = _batch
            batch_non_pad_mask=batch_non_pad_mask.to(torch.uint8)
            # thinning can only run in single instance mode, not in batch mode
            num_batch = time_delta_seq.size(0)
            for i in range(num_batch):

                rst = []
                _time_delta_seq, _event_seq = time_delta_seq[i][batch_non_pad_mask[i]], event_seq[i][batch_non_pad_mask[i]]
                seq_len = _time_delta_seq.size(0)
                duration = _time_delta_seq[seq_len-2].item() + np.finfo(float).eps
                num_sub = seq_len - 1
                
                ### not use len 
                next_event_name, next_event_time = _event_seq[seq_len-1].item(), _time_delta_seq[seq_len-1].item()
                current_event_name, current_event_time = _event_seq[seq_len-2].item(), _time_delta_seq[seq_len-2].item()
                #time_last_event = _time_delta_seq[seq_len-2].item()
                #if verbose:
                #        print(f"for {seq_id}-th seq, predict after seq_len-1-th event {current_event_name} at {current_event_time:.4f}")
                    
                #next_event_dtime = next_event_time# - time_last_event
                #avg_future_dtime = _time_delta_seq.mean()#最後のイベント時刻-lenごと直近の経過時間 / 残りの予測対象数
                #look_ahead = max(torch.tensor(next_event_dtime), avg_future_dtime)#次のイベントまでか、平均発生時刻の大きいほう
                boundary = torch.tensor(self.max_tau*1.5)#基準の4倍＋予測対象
                
                boundary=boundary.cuda()
                
                #予測対称がjとすると、イベントの種類履歴まで, 経過時間の履歴まで
                _event_prefix, _time_prefix = _event_seq[:seq_len-1].unsqueeze(0).cuda(), _time_delta_seq[:seq_len-1].unsqueeze(0).cuda()
                accepted_times, weights = thinning_sampler.draw_next_time(
                    [[_event_prefix, _time_prefix],
                     boundary, model]
                )#予測時間、重み=[履歴]、直近イベント、境界、モデル
                
                time_uncond = float(torch.sum(accepted_times * weights))#重み*アクセプト時刻の総和=未知の時間
                #dtime_uncond = time_uncond - time_last_event#未知の時刻ー最後の時刻＝未知の経過時間
                intensities_at_times = model.compute_intensities_at_sampled_times(
                    _event_prefix, _time_prefix,
                    _time_delta_seq[seq_len-1].reshape(1, 1).cuda()
                )[0, 0]#その時刻での強度=model(イベント種類履歴、経過時刻履歴、経過時間)
                top_ids = torch.argsort(intensities_at_times, dim=0, descending=True)#種類ごとの強度のトップいくつか
                # since we use int to represent event names already
                top_event_names = [int(top_i) for top_i in top_ids]#topのイベント種類番号
                rst.append(
                    (
                        time_uncond, top_event_names,
                        next_event_time, next_event_name
                    )
                )
                se_his.append(((time_uncond-current_event_time)**2))
                pred_his.append(time_uncond)
                GT_his.append(current_event_time)
                if verbose:
                    print(
                        f"our predicted time is {time_uncond:.4f} and sorted event types are :\n{top_event_names}")
                    print(
                        f"gold ({next_event_name}) ranked {top_event_names.index(next_event_name)} out of {len(top_event_names)}")
                """
                for j in range(seq_len - 1):
                    # next_event_name, next_event_time = _event_seq[j + 1].item(), _time_delta_seq[j + 1].item()
                    # current_event_name, current_event_time = _event_seq[j].item(), _time_delta_seq[j].item()
                    # time_last_event = _time_delta_seq[j].item()
                    if verbose:
                        print(f"for {seq_id}-th seq, predict after {j}-th event {current_event_name} at {current_event_time:.4f}")
                    
                    next_event_dtime = next_event_time# - time_last_event
                    avg_future_dtime = (duration - time_last_event) / (num_sub - j)#最後のイベント時刻-lenごと直近の経過時間 / 残りの予測対象数
                    look_ahead = max(next_event_dtime, avg_future_dtime)#次のイベントまでか、平均発生時刻の大きいほう
                    boundary = time_last_event + 4 * look_ahead#基準の4倍＋予測対象
                    
                    #予測対称がjとすると、イベントの種類履歴まで, 経過時間の履歴まで
                    _event_prefix, _time_prefix = _event_seq[:j + 1].unsqueeze(0).cuda(), _time_delta_seq[:j + 1].unsqueeze(0).cuda()
                    accepted_times, weights = thinning_sampler.draw_next_time(
                        [[_event_prefix, _time_prefix],
                        time_last_event, boundary, model]
                    )#予測時間、重み=[履歴]、直近イベント、境界、モデル
                    
                    time_uncond = float(torch.sum(accepted_times * weights))#重み*アクセプト時刻の総和=未知の時間
                    #dtime_uncond = time_uncond - time_last_event#未知の時刻ー最後の時刻＝未知の経過時間
                    intensities_at_times = model.compute_intensities_at_sampled_times(
                        _event_prefix, _time_prefix,
                        _time_delta_seq[j + 1].reshape(1, 1).cuda()
                    )[0, 0]#その時刻での強度=model(イベント種類履歴、経過時刻履歴、経過時間)
                    top_ids = torch.argsort(intensities_at_times, dim=0, descending=True)#種類ごとの強度のトップいくつか
                    # since we use int to represent event names already
                    top_event_names = [int(top_i) for top_i in top_ids]#topのイベント種類番号
                    rst.append(
                        (
                            time_uncond, top_event_names,
                            next_event_time, next_event_dtime, next_event_name
                        )
                    )
                    if verbose:
                        print(
                            f"our predicted time is {time_uncond:.4f} and sorted event types are :\n{top_event_names}")
                        print(
                            f"gold ({next_event_name}) ranked {top_event_names.index(next_event_name)} out of {len(top_event_names)}")
                """
                results.append(rst)
                seq_id += 1
        se=np.array(se_his).mean()
        std=np.sqrt(np.std(se_his))
        print(f"se:{se}, rmse:{np.sqrt(se)},std:{std}")
        return results,np.sqrt(se),GT_his,pred_his,std



