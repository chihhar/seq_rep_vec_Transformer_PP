# -*- coding: utf-8 -*-
# !/usr/bin/python
import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import argparse
__author__ = 'Hongyuan Mei, Chenghao Yang'


class EventSampler(nn.Module):
    """
    sample one event from a (structure-)NHP using chosen algorithm
    """
    def __init__(self, num_sample, num_exp, device=None):
        super(EventSampler, self).__init__()
        self.num_sample = num_sample#100
        self.num_exp = num_exp#500
        device = device or 'cpu'
        self.device = torch.device(device)
        self.patience_counter = 0

    def cuda(self, device=None):
        device = device or 'cuda:0'
        self.device = torch.device(device)
        assert self.device.type == 'cuda'
        super().cuda(self.device)

    def cpu(self):
        self.device = torch.device('cpu')
        super().cuda(self.device)

    def draw_next_time(self, arguments, mode='ordinary'):
        if mode=='ordinary':
            rst, weights = self.draw_next_time_ordinary(arguments)
        elif mode=='fractional':
            raise NotImplementedError
        elif mode == "unbiased":
            rst, weights = self.draw_next_time_unbiased(arguments)
        elif mode == "log2":
            rst, weights = self.draw_next_time_unbiased(arguments)
        else:
            raise Exception(f"Unknow sampling mode : {mode}")
        return rst, weights

    def draw_next_time_ordinary(self, arguments, fastapprox=True):
        #batch, time_last_event, boundary, datalog = arguments
        pdb.set_trace()
        batch, boundary, datalog = arguments#[[29このイベントタイプ履歴,29個の履歴time],boundary,model] boundaryは想定最大値
        event_seq, time_seq = batch
        assert event_seq.size(0) == 1, "Currently, thinning algorithm do not support batch-ed computation"
        """
        ordinary thinning algorithm (with a little minor approximation)
        """
        """
        sample some time points to get intensities 
        s.t. we can estimate a conservative upper bound
        """
        over_sample_rate = 5.0
        times_for_bound = torch.empty(
            size=[10], dtype=torch.float32, device=self.device
        ).uniform_( 0, boundary )#shape[10]
        intensities_for_bound = datalog.compute_intensities_at_sampled_times(event_seq,
                                                            time_seq,
                                                            times_for_bound.unsqueeze(0)
                                                            ).squeeze(0) #最大値でのintensity[10,types=1]
        bounds = intensities_for_bound.sum(dim=-1).max() * over_sample_rate#1個　tensor(0.3042, device='cuda:0', grad_fn=<MulBackward0>)
        sample_rate = bounds
        """
        estimate # of samples needed to cover the interval
        """
        """
        
        bound is roughly C * intensity (C>1), so accept rate is 1/C
        if we have N samples, then prob of no accept is only (1 - 1/C)^N
        meaning that if we want accept at least one sample with 99% chance
        we need > log0.01 / log(1 - 1/C) samples 
        if we make C reasonable small, e.g., 
        C = 2, # of samples : 6
        C = 5, # of samples : 20
        C = 10, # of samples : 44
        therefore, some reasonable num_exp is good enough, e.g, 100 or 500
        if we use 100 samples, for each C, prob(at one sample) 
        C = 2, 99.99%
        C = 5, 99.99%
        C = 10, 99.99%
        a benefit of using large S : making sure accumulated times reach boundary
        in that case, if none is accepted, use the farthest time as prediction
        """
        S = self.num_exp # num_exp is usually large enough for 2 * intensity bound  =500
        """
        prepare result
        """
        rst = torch.empty(
            size=[self.num_sample], dtype=torch.float32, device=self.device
        ).fill_(boundary)#[100]のboundaryが埋まっただけ
        # for those didn't accept proposed times, use boundary or even farther
        weights = torch.ones(
            size=[self.num_sample], dtype=torch.float32, device=self.device)
        weights /= weights.sum()#全体が1になるように正規化。（等分）
        """
        sample times : dt ~ Exp(sample_rate) #??指数で微分?
        compute intensities at these times # sampleでの強度を計算
        """
        if fastapprox:#True
            """
            reuse the proposed times to save computation (of intensities)　# 提案された時刻を計算量を保存するために再利用。
            different draws only differ by which of them is accepted #～と異なるのは採用されるのが違うくらい。
            but proposed times are same #ただ提案される時間は同様。
            """
            Exp_numbers = torch.empty(
                size=[1, S], dtype=torch.float32, device=self.device)
        else:
            Exp_numbers = torch.empty(
                size=[self.num_sample, S], dtype=torch.float32, device=self.device)
        Exp_numbers.exponential_(1.0)#指数分布 S=500個
        sampled_times = Exp_numbers / sample_rate#[1,500]
        sampled_times = sampled_times.cumsum(dim=-1) #累積で時間を作っている。
        intensities_at_sampled_times = datalog.compute_intensities_at_sampled_times(event_seq, time_seq, sampled_times)#[1,500,1]
        total_intensities = intensities_at_sampled_times.sum(dim=-1)#[1,500] typesをまとめた
        # size : N * S or 1 * S
        # datalog.clear_cache() # clear cache of embeddings
        if fastapprox:
            """
            reuse proposed times and intensities at those times
            """
            sampled_times = sampled_times.expand(self.num_sample, S)#[100,500]
            total_intensities = total_intensities.expand(self.num_sample, S)#[100,500]
        """
        randomly accept proposed times
        """
        Unif_numbers = torch.empty(
            size=[self.num_sample, S], dtype=torch.float32, device=self.device)
        Unif_numbers.uniform_(0.0, 1.0)#0~1の一様分布[100,500]
        criterion = Unif_numbers * sample_rate / total_intensities#[100,500]
        """
        for each parallel draw, find its min criterion #最小の基準
        if that < 1.0, the 1st (i.e. smallest) sampled time with cri < 1.0 is accepted 
        if none is accepted, use boundary/maxsampletime for that draw
        """
        min_cri_each_draw, _ = criterion.min(dim=1)#[100] 500個のうち最小のもの
        who_has_accepted_times = min_cri_each_draw < 1.0#全てTrue
        """
        whoever accepts times, find their accepted times
        """
        sampled_times_accepted = sampled_times.clone()#[100,500]の実態
        sampled_times_accepted[criterion>=1.0] = sampled_times.max() + 1.0#criterion>=1.0の部分にsampleの最大値+1
        accepted_times_each_draw, accepted_id_each_draw = sampled_times_accepted.min(dim=-1)
        #accepted_times_each_draw:[100]
        #accepted_id_each_draw 500次元方向での最小値idが分かる
        # size : N
        #rstは強度の仮の最大値[100]サンプル分用意したもの
        rst[who_has_accepted_times] = \
            accepted_times_each_draw[who_has_accepted_times]#[100],サンプル100個のうち全部が使える[True] すべてを１00こサンプルのacceptの最小で割る。rstもし使えなかったら、boundaryが埋まっているだけ。
        who_not_accept = ~who_has_accepted_times#acceptされない場所。　などない。
        who_reach_further = sampled_times[:, -1] > boundary#全部boundaryよりもでかい。
        rst[who_not_accept&who_reach_further] = \
            sampled_times[:, -1][who_not_accept&who_reach_further]#何の処理も行われない
        return rst, weights

    # utils func for adaptive thinning
    def find_bounds(self, arguments):
        batch, time_last_event, boundary, datalog = arguments
        event_seq, time_seq = batch
        assert event_seq.size(0) == 1, "Currently, thinning algorithm do not support batch-ed computation"
        over_sample_rate = 5.0
        times_for_bound = torch.empty(
            size=[10], dtype=torch.float32, device=self.device
        ).uniform_( time_last_event, boundary )
        intensities_for_bound = datalog.compute_intensities_at_sampled_times(event_seq,
                                                                             time_seq,
                                                                             times_for_bound.unsqueeze(0)
                                                                             ).squeeze(0)
        bounds = intensities_for_bound.sum(dim=-1).max() * over_sample_rate
        return bounds

    def compute_bounds(self, arguments, num_samples=10, over_sample_rate=5.0):
        batch, time_last_event, boundary, datalog = arguments
        event_seq, time_seq = batch
        assert event_seq.size(0) == 1, "Currently, thinning algorithm do not support batch-ed computation"
        times_for_bound = torch.empty(
            size=[num_samples], dtype=torch.float32, device=self.device
        ).uniform_( time_last_event, boundary )
        intensities_for_bound = datalog.compute_intensities_at_sampled_times(event_seq,
                                                                             time_seq,
                                                                             times_for_bound.unsqueeze(0)
                                                                             ).squeeze(0)

        bounds = intensities_for_bound.sum(dim=-1).max() * over_sample_rate
        return bounds

    def draw_next_time_unbiased(self, arguments):
        # implement adaptive thinning, approximately unbiased -- except in the update of LB and RB
        batch, time_last_event, boundary, datalog, patience  = arguments
        event_seq, time_seq = batch
        assert event_seq.size(0) == 1, "Currently, thinning algorithm do not support batch-ed computation"
        assert self.num_sample == 1, "Currently, only support N=1 -- will write N>1 in the future"
        rst = torch.empty(
            size=[self.num_sample], dtype=torch.float32, device=self.device
        ).fill_(boundary)
        weights = torch.ones(
            size=[self.num_sample], dtype=torch.float32, device=self.device)
        weights /= weights.sum()
        target_num_samples = self.num_sample
        _arguments = [batch, time_last_event, boundary, datalog]
        look_ahead = boundary - time_last_event
        num_samples_for_bounds = 100
        while target_num_samples > 0 and patience > 0:
            bounds = self.compute_bounds(_arguments, num_samples=num_samples_for_bounds, over_sample_rate=2)
            sample_rate = bounds
            S = self.num_exp
            Exp_numbers = torch.empty(size=[target_num_samples, S], dtype=torch.float32, device=self.device)
            Exp_numbers.exponential_(1.0)
            sampled_times = Exp_numbers / sample_rate
            sampled_times = sampled_times.cumsum(dim=-1) + time_last_event

            min_time_per_row, _ = sampled_times.min(dim=-1)
            accepted_rows = min_time_per_row < boundary
            sampled_times = sampled_times[accepted_rows]

            intensities_at_sampled_times = datalog.compute_intensities_at_sampled_times(event_seq, time_seq,
                                                                                        sampled_times)
            total_intensities = intensities_at_sampled_times.sum(dim=-1)
            cur_sample_num = sampled_times.size(0)
            Unif_numbers = torch.empty(
                size=[cur_sample_num, S], dtype=torch.float32, device=self.device)
            Unif_numbers.uniform_(0.0, 1.0)
            criterion = Unif_numbers * sample_rate / total_intensities
            # no need to consider out-of-the-boundary examples
            criterion[sampled_times >= boundary] = 1.1
            min_cri_each_draw, _ = criterion.min(dim=1)
            who_has_accepted_times = min_cri_each_draw < 1.0
            sampled_times_accepted = sampled_times.clone()
            sampled_times_accepted[criterion >= 1.0] = sampled_times.max() + 1.0
            accepted_times_each_draw, accepted_id_each_draw = sampled_times_accepted.min(dim=-1)
            accepted_sample_num_this_round = min(target_num_samples, who_has_accepted_times.sum())
            if accepted_sample_num_this_round > 0:
                if accepted_sample_num_this_round < target_num_samples:
                    rst[-target_num_samples: -target_num_samples + accepted_sample_num_this_round] = \
                        accepted_times_each_draw[who_has_accepted_times][:accepted_sample_num_this_round]
                else:
                    rst[-target_num_samples: ] = \
                        accepted_times_each_draw[who_has_accepted_times][:accepted_sample_num_this_round]


            time_last_event += min(sampled_times.max().item() - time_last_event, look_ahead)
            boundary = time_last_event + look_ahead
            target_num_samples -= accepted_sample_num_this_round
            if accepted_sample_num_this_round == 0:
                patience -= 1
            if target_num_samples > 0:
                rst[-target_num_samples:] = boundary


            _arguments = [batch, time_last_event, boundary, datalog]

        if patience == 0:
            self.patience_counter += 1

        return rst, weights

