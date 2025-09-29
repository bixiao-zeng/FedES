import torch
import argparse
import sys
import os
from typing import Dict, Tuple, List, Optional

from fednoisy.data.NLLData import functional as nllF

import subprocess
import pandas as pd
from io import StringIO,BytesIO
import distutils

def read_fednll_args():
    parser = argparse.ArgumentParser(description="Federated Noisy Labels Preparation")

    # ==== Pipeline args ====

    parser.add_argument(
        "--num_clients",
        default=20,
        type=int,
        help="Number for clients in federated setting.",
    )
    parser.add_argument("--com_round", type=int, default=3)
    parser.add_argument(
        "--model",
        type=str,
        default="ResNet18",
        help="Currently only support 'Cifar10Net', 'SimpleCNN',  'LeNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'ToyModel', 'ResNet18', 'PreResNet18', 'ResNet20', 'WRN28_10', 'WRN40_2' and 'ResNet34'.",
    )
    parser.add_argument("--sample_ratio", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)

    # ==== FedNLL data args ====
    parser.add_argument(
        "--centralized",
        default=False,
        help="Centralized setting or federated setting. True for centralized "
        "setting, while False for federated setting.",
    )
    parser.add_argument(
        "--preload", action="store_true", help="Whether to preload dataset into memory."
    )
    # ----Federated Partition----
    parser.add_argument(
        "--partition",
        default="noniid-labeldir",
        type=str,
        choices=["iid", "noniid-#label", "noniid-labeldir", "noniid-quantity",'real'],
        help="Data partition scheme for federated setting.",
    )

    parser.add_argument(
        "--dir_alpha",
        default=0.5,
        type=float,
        help="Parameter for Dirichlet distribution.",
    )
    parser.add_argument(
        "--major_classes_num",
        default=2,
        type=int,
        help="Major class number for 'noniid-#label' partition.",
    )
    parser.add_argument(
        "--min_require_size",
        default=10,
        type=int,
        help="Minimum sample size for each client.",
    )
    parser.add_argument(
        "--process_method",
        type=str,
        default="raw",
        help="the feature process method",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="mel_spec",
        help="the feature type",
    )

    # ----Noise setting options----
    parser.add_argument(
        "--noise_mode",
        default='norm',
        type=str,
        help='uniform distribution for a defined noise spread, beta distribution for a extreme noise spread, real refers to real-world noise'
    )
    
    parser.add_argument(
        "--asym_rate",
        default=0,
        type=float,
        help='the ratio of asymmetric noisy clients to all clients'
    )
    parser.add_argument(
        "--mean",
        default=0.5,
        type=float,
        help="mean for normal noise ratio",
    )
    parser.add_argument(
        "--std",
        default=0.2,
        type=float,
        help="std for normal noise ratio.",
    )
    
    
    parser.add_argument(
        "--num_samples",
        default=32 * 2 * 1000,
        type=int,
        help="Number of samples used for Clothing1M training. Defaults as 64000.",
    )

    # ----Robust Loss Function options----
    parser.add_argument(
        "--criterion", type=str, default="ce"
    )  # for robust loss function
    parser.add_argument(
        "--sce_alpha",
        type=float,
        default=0.1,
        help="Symmetric cross entropy loss: alpha * CE + beta * RCE",
    )
    parser.add_argument(
        "--sce_beta",
        type=float,
        default=1.0,
        help="Symmetric cross entropy loss: alpha * CE + beta * RCE",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=1.0,
        help="scale parameter for loss, for example, scale * RCE, scale * NCE, scale * normalizer * RCE.",
    )
    parser.add_argument(
        "--gce_q",
        type=float,
        default=0.7,
        help="q parametor for Generalized-Cross-Entropy, Normalized-Generalized-Cross-Entropy.",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=None,
        help="alpha parameter for Focal loss and Normalzied Focal loss.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=0.0,
        help="gamma parameter for Focal loss and Normalzied Focal loss.",
    )
    # ----Shared options----
    parser.add_argument('--warm', type=int, default=1)
    parser.add_argument('--LA', type=int, default=1)
    parser.add_argument('--begin', type=int, default=0, help='2nd stage start from which round')
    parser.add_argument('--end', type=int, default=100, help='2nd stage end at which round')

    parser.add_argument("--favg", action="store_true", help="Whether to use FedAvg.")

    # ----FedTD options----
    parser.add_argument("--td", action="store_true", help="Whether to use FedTD.")
    parser.add_argument("--sudo", type=int, default=100, help="When to use sudo label for symmetric label nosie.")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature of early distillation.")

    # ----Mixup options----
    parser.add_argument("--mixup", action="store_true", help="Whether to use mixup.")
    parser.add_argument(
        "--mixup_alpha", type=float, default=1.0, help="Hyperparameter alpha for mixup."
    )


    # ----Watcher options----
    parser.add_argument("--watch", action="store_true", help="Whether to use watcher.")
    parser.add_argument("--memory", action="store_true", help="Whether to use memory.")
    parser.add_argument("--watchtst", action="store_true", help="Whether to watch test error on different type of noisy client.")
    parser.add_argument("--watchtr", action="store_true", help="Whether to watch test error on different type of noisy client.")
    parser.add_argument("--watchtrloss", action="store_true", help="Whether to watch test error on different type of noisy client.")


    # parser.add_argument("--Dagg", default=True, type=lambda x:bool(distutils.util.strtobool(x)),help="Whether to use Dagg from noro.")
    parser.add_argument("--Dagg", action="store_true",help="Whether to use Dagg from noro.")
    parser.add_argument("--denoi_noro", action="store_true", help="Whether to use denoise_noro from noro.")
    parser.add_argument("--denoi_sce", action="store_true", help="Whether to use denoise_sce from sce.")
    parser.add_argument("--clean_sce", action="store_true", help="Whether to use cross entropy for clean clients.")
    parser.add_argument("--denoi_oldsce", action="store_true", help="Whether to use denoise_oldsce from sce.")
    parser.add_argument("--estsce_tscse", action="store_true", help="Whether to use estsce_tscse.")
    parser.add_argument("--esceOnly", action="store_true", help="Whether to use early-stopping cross-entropy only.")
    parser.add_argument("--es2tsce_tscse", action="store_true", help="Whether to use estsce_tscse.")
    parser.add_argument("--real_sce_noro", action="store_true", help="Whether to use real_sce_noro.")
    parser.add_argument("--real_ESsce_sce", action="store_true", help="Whether to use real_sce_noro.")
    parser.add_argument("--real_tsce_sce", action="store_true", help="Whether to use tsce for asym and sce for sym.")
    parser.add_argument("--wu_watch", action="store_true", help="Whether to use warmup_watch.")
    parser.add_argument("--warm_LA", type=int, default=0,help="Whether to use warm LA.")
    parser.add_argument("--cold_LA", type=int, default=1,help="Whether to use cold LA.")

    # ----CDR options----
    parser.add_argument("--cdr", action="store_true", help="Whether to use cdr.")
    parser.add_argument("--realq", action="store_true", help="Whether to use cdr_realq.")
    parser.add_argument("--sgn", action="store_true", help="Whether to use sgn for EMD.")
    parser.add_argument("--dynamic", action="store_true", help="Whether to use dynamic estimation of noise rate.")
    parser.add_argument("--stage1", type=int, default=0,help="How many rounds the first stage uses")
    parser.add_argument("--plt_metric", action="store_true", help="Whether to use dynamic estimation of noise rate.")
    parser.add_argument("--load_metric", action="store_true", help="Whether to use dynamic estimation of noise rate.")
    parser.add_argument("--grad", action="store_true", help="Whether to use single grad to denote parameter criticality.")

    # ----NoRo options----
    parser.add_argument("--noro", action="store_true", help="Whether to use noro.")
    parser.add_argument('--a', type=float, default=0.8, help='a')
    parser.add_argument('--oldclntsel', type=int, default=1,help='use old version client selection')
    parser.add_argument("--bk_rnd", type=int,default=-1, help="start from the beginning of which round.")

    # ----Corr options----
    parser.add_argument("--corr", action="store_true", help="Whether to use FedCorr.")
    parser.add_argument("--corroffi", action="store_true", help="Whether to use FedCorr official version.")
    parser.add_argument('--iteration1', type=int, default=5, help="enumerate iteration in preprocessing stage")
    parser.add_argument('--rounds1', type=int, default=200, help="rounds of training in fine_tuning stage")
    parser.add_argument('--rounds2', type=int, default=200, help="rounds of training in usual training stage")
    parser.add_argument('--frac1', type=float, default=0.01, help="fration of selected clients in preprocessing stage")
    parser.add_argument('--frac2', type=float, default=0.1,
                        help="fration of selected clients in fine-tuning and usual training stage")
    parser.add_argument('--relabel_ratio', type=float, default=0.5,
                        help="proportion of relabeled samples among selected noisy samples")
    parser.add_argument('--confidence_thres', type=float, default=0.5,
                        help="threshold of model's confidence on each sample")
    parser.add_argument('--clean_set_thres', type=float, default=0.1,
                        help="threshold of estimated noise level to filter 'clean' set used in fine-tuning stage")
    parser.add_argument('--fedmixup', action='store_true')
    parser.add_argument('--beta', type=float, default=5, help="coefficient for local proximal，0 for fedavg, 1 for fedprox, 5 for noise fl")
    parser.add_argument('--correction', type=int, default=1, help="whether use label correction")






    # ----RoFL options----
    parser.add_argument("--rofl", action="store_true", help="Whether to use rofl.")
    parser.add_argument('--T_pl', type=int, help='T_pl: When to start using global guided pseudo labeling', default=100)
    parser.add_argument('--lambda_cen', type=float, help='lambda_cen', default=1.0)
    parser.add_argument('--lambda_e', type=float, help='lambda_e', default=0.8)
    parser.add_argument('--feature_dim', type=int, help = 'feature dimension', default=512)
    parser.add_argument('--num_gradual', type=int, default=10, help='T_k')
    parser.add_argument('--forget_rate', type=float, default=0.2, help="forget rate")


    # ----Co-teaching options----
    parser.add_argument(
        "--coteaching", action="store_true", help="Whether to use co-teahcing."
    )
    parser.add_argument(
        "--coteaching_forget_rate",
        type=float,
        default=None,
        help="Forget rate for co-teaching.",
    )
    parser.add_argument(
        "--coteaching_exponent",
        type=float,
        default=1,
        help="exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.",
    )
    parser.add_argument(
        "--coteaching_num_gradual",
        type=int,
        default=25,
        help="how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.",
    )

    # ----Dynamic Bootstrapping options----
    parser.add_argument(
        "--dynboot",
        action="store_true",
        help="Whether to use Dynamic Bootstrapping. Original paper is 'Unsupervised Label Noise Modeling and Loss Correction'.",
    )
    parser.add_argument(
        "--dynboot_mixup",
        type=str,
        default="static",
        choices=["static", "dynamic"],
        help="Dynamic Bootstrapping: Type of bootstrapping. Available: 'static' (as in the paper, default), 'dynamic' (BMM to mix the smaples, will use decreasing softmax). Default: 'static'",
    )
    # parser.add_argument("--debug", action="store_true")
    # parser.add_argument(
    #     "--dynboot_M",
    #     nargs="+",
    #     type=int,
    #     default=[167, 417],
    #     help="Milestones for the LR sheduler, default 100 250",
    # )
    parser.add_argument(
        "--dynboot_alpha",
        type=float,
        default=32,
        help="Dynamic Bootstrapping: alpha parameter for the mixup distribution, default: 32",
    )
    parser.add_argument(
        "--dynboot_bootbeta",
        type=str,
        default="hard",
        choices=[None, "hard", "soft"],
        help="Dynamic Bootstrapping: Type of Bootstrapping guided with the BMM. Available: \
                        None (deactivated)(default), 'hard' (Hard bootstrapping), 'soft' (Soft bootstrapping), default: 'hard'",
    )
    parser.add_argument(
        "--dynboot_reg",
        type=float,
        default=0.0,
        help="Dynamic Bootstrapping: Parameter of the regularization term, default: 0.",
    )

    # ----Path options----
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        choices=["mnist", "cifar10", "cifar100", "svhn", "clothing1m", "webvision",'gcommand'],
        help="Dataset for experiment. Current support: ['mnist', 'cifar10', "
        "'cifar100', 'svhn', 'clothing1m', 'webvision']",
    )
    parser.add_argument(
        "--data_dir",
        default="./fedNLLdata/cifar10",
        type=str,
        help="Directory to save the dataset with noisy labels.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./Fed-Noisy-checkpoint/cifar10",
        help="Checkpoint path for log files and report files.",
    )

    # ----Miscs options----

    parser.add_argument(
        "--save_best", action="store_true", help="Whether to save the best model."
    )
    parser.add_argument(
        "--load_last", action="store_true", help="Whether to load the last model."
    )
    parser.add_argument(
        "--start_round", type=int, help="the start round of corrupted training"
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed")

    args = parser.parse_args()
    return args

def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(BytesIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
    idx = pd.to_numeric(gpu_df['memory.free']).idxmax()
    print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx