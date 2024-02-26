import sys
sys.path.append("../")
import pickle
from matplotlib import pyplot as plt
import time
import numpy
import random
import os
import datetime

import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from esm.tester import Tester
from utils.log import LogReader
import numpy as np
import pdb
import argparse
sys.path.append("../")
call1s_path="/data1/nishizawa/anhp-andtt/domains/call1/ContKVLogs/h-8_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-18330/"
call2s_path="/data1/nishizawa/anhp-andtt/domains/call2/ContKVLogs/h-8_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-24649/"
call3s_path="/data1/nishizawa/anhp-andtt/domains/call3/ContKVLogs/h-8_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-12400/"

def main():
    #call1 load
    is_file = os.path.isfile("anhp-andtt/domains/call1/ContKVLogs/h-8_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-18330/results_testpred_his")
    if is_file:
        print(f"{path} is a file.")
    else:
        print("nothing") # パスが存在しないかファイルではない
    is_dir = os.path.isdir("../../")
    if is_dir:
        print(f"{path} is a directory.")
    else:
        pass # パスが存在しないかディレクトリではない
    with open("anhp-andtt/domains/call1/ContKVLogs/h-8_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-18330/results_testpred_his", 'rb') as f:
        call1pred_his = pickle.load(f)
    with open("anhp-andtt/domains/call1/ContKVLogs/h-8_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-18330/results_testGT_his", 'rb') as f:
        call1GT_his = pickle.load(f)
    with open("anhp-andtt/domains/call1/ContKVLogs/h-8_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-18330/results_testpred_his", 'rb') as f:
        call1pred_his = pickle.load(f)
    with open(call2s_path+"reusults_testGT_his", 'rb') as f:
        call2GT_his = pickle.load(f)
    with open(call2s_path+"reusults_testpred_his", 'rb') as f:
        call2pred_his = pickle.load(f)
    with open(call3s_path+"reusults_testGT_his", 'rb') as f:
        call3GT_his = pickle.load(f)
    with open(call3s_path+"reusults_testpred_his", 'rb') as f:
        call3pred_his = pickle.load(f)
    print(call1GT_his)

if __name__ == "__main__":
    plt.switch_backend('agg')
    plt.figure()
    main()