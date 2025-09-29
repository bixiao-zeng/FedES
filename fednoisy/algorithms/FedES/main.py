#===========search and define the optimal GPU=======================
import torch

import subprocess
import pandas as pd
from io import StringIO,BytesIO
# def get_free_gpu():
#     gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
#     gpu_df = pd.read_csv(BytesIO(gpu_stats),
#                          names=['memory.used', 'memory.free'],
#                          skiprows=1)
#     print('GPU usage:\n{}'.format(gpu_df))
#     gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
#     idx = pd.to_numeric(gpu_df['memory.free']).idxmax()
#     print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
#     return idx

# torch.cuda.set_device(free_gpu_id)
# # args.device = torch.device("cuda:{}".format(free_gpu_id))  # Use GPU 2
# # print('recommend device',str(free_gpu_id))
print('current device',torch.cuda.current_device())
torch.cuda.synchronize()

from json import load
import os
import sys
import argparse
import random
from copy import deepcopy

from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from fedlab.utils.logger import Logger
from fedlab.utils.aggregator import Aggregators

sys.path.append(os.getcwd())
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
    CIFAR10_TRANSITION_MATRIX,
    NORM_VALUES,
)
from fednoisy.data.NLLData import functional as nllF
from fednoisy.algorithms.FedES.client import (
    FedNLLFedAvgClientTrainer,
FedNLLFedAvgESClientTrainer
)
from fednoisy.algorithms.FedES.server import FedAvgServerHandler,FedAvgESServerHandler

from fednoisy.algorithms.FedES.standalone import FedAvgStandalone,FedAvgESStandalone
from fednoisy.algorithms.FedES.misc import read_fednll_args,get_free_gpu
from fednoisy.data.dataset import FedNLLDataset
from fednoisy.utils.misc import (
    setup_seed,
    make_dirs,
    make_exp_name,
    result_parser,
    make_alg_name,
)
from fednoisy.models.build_model import build_model, build_multi_model
import numpy as np

list1 = [1,2,3,4]
mean = np.mean(list1)

args = read_fednll_args()

if torch.cuda.is_available():
    args.cuda = True
else:
    args.cuda = False

setup_seed(args.seed)
if args.dataset == "clothing1m":
    args.noise_mode = "real"
    args.globalize = True
    args.noise_ratio = 0.39

nll_name = nllF.FedNLL_name(**vars(args))
exp_name = make_exp_name("fedavg", args)
alg_name = make_alg_name(args)
cmp_out_dir = os.path.join(args.out_dir, nll_name, alg_name, exp_name)
make_dirs(cmp_out_dir)



if args.coteaching is True:
    model = build_multi_model(
        args.model, CLASS_NUM[args.dataset], dataset=args.dataset, num_models=2
    )
else:
    model = build_model(args,args.model, CLASS_NUM[args.dataset], dataset=args.dataset)

# ==== prepare logger from fedlab====
# server_logger = Logger(
#     log_name="ServerHandler",
#     log_file=os.path.join(cmp_out_dir, "server.log"),
# )
#
# client_logger = Logger(
#     log_name="ClientTrainer",
#     log_file=os.path.join(cmp_out_dir, "client.log"),
# )

# ==== prepare logger from loguru====
server_logger = os.path.join(cmp_out_dir, "server.log")

client_logger = os.path.join(cmp_out_dir, "client.log")
print(cmp_out_dir)
# ==== choose server handler and client trainer ====

if args.favg:
    trainer = FedNLLFedAvgClientTrainer(model, args.num_clients, cuda=True, loggerfile=client_logger, args=args)
    handler = FedAvgServerHandler(
    model, args.com_round, args.sample_ratio, cuda=True, loggerfile=server_logger, args=args
)  # server
    pipeline = FedAvgStandalone(handler, trainer, args=args, save_best=args.save_best,save_last=True)
else:
    trainer = FedNLLFedAvgESClientTrainer(model, args.num_clients, cuda=True, loggerfile=client_logger, args=args)
    handler = FedAvgESServerHandler(
    model, args.com_round, args.sample_ratio, cuda=True, loggerfile=server_logger, args=args
)  # server
    pipeline = FedAvgESStandalone(handler, trainer, args=args, save_best=args.save_best,save_last=True)




# ==== server dataset ====
handler_dataset = FedNLLDataset(args, test_preload=args.preload)
handler.setup_dataset(handler_dataset)

# ==== client trainer dataset ====
trainer_dataset = FedNLLDataset(
    args, train_preload=args.preload, test_preload=args.preload
)
trainer.setup_dataset(trainer_dataset)
trainer.setup_optim(
    args.epochs, args.batch_size, args.lr, args.weight_decay, args.momentum
)

# ====  launch pipeline ====
print(f"FedNLL scene: {nll_name}")
import torch




    
pipeline.main()
