import copy
import pickle

import torch
import argparse
import sys
import os
import numpy as np
from copy import deepcopy
from typing import Dict, Tuple, List, Optional

from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from scipy.stats import entropy
from collections import defaultdict

from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer

import torch.nn.functional as F
from fednoisy.data.NLLData import functional as nllF
from fednoisy.data.dataset import DatasetRoFL
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
    CIFAR10_TRANSITION_MATRIX,
    NORM_VALUES,
)
from fednoisy.utils.misc import (
    setup_seed,
    make_dirs,
    make_exp_name,
    result_parser,
    make_alg_name,
    AverageMeter,
    WatchSystematicNoise,
)
from fednoisy.utils import misc as misc
from fednoisy.utils.criterion import get_robust_loss, mixup_criterion, loss_coteaching,SCELoss,LA_SCELoss,LA_SCELoss_kd
from fednoisy.utils.mixup import mixup_data
from fednoisy.utils import dynamic_bootstrapping as dynboot

from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from sklearn.decomposition import PCA,IncrementalPCA,TruncatedSVD
from scipy.spatial.distance import cdist

sys.path.append(os.getcwd())
print(os.getcwd())



class FedNLLFedAvgClientTrainer(SGDSerialClientTrainer):
    def __init__(
        self,
        model,
        num_clients,
        cuda=True,
        device=None,
        loggerfile=None,
        personal=False,
        args=None,
    ) -> None:
        SGDSerialClientTrainer.__init__(
            self, model, num_clients, cuda, device, logger, personal
        )
        self.cur_cid = None
        self.cache = []
        self.args = args
        writer = SummaryWriter(comment='scalar')
        logger.add(loggerfile, filter=lambda record: record['extra'].get('name') == 'c')
        self.logger = logger.bind(name='c')
        self.logger.debug(writer.log_dir.replace('runs/', ''))
        self.watcher = WatchSystematicNoise(self.args.num_clients)
        nll_name = nllF.FedNLL_name(**vars(args))
        exp_name = make_exp_name("fedavg", args)
        alg_name = make_alg_name(args)
        self.cmp_out_dir = os.path.join(args.out_dir, nll_name, alg_name, exp_name)

        self.nll_name = nllF.FedNLL_name(**vars(args))
        self.estimate_quality = np.ones(args.num_clients)
        nll_filename = f"{self.nll_name}_seed_{args.seed}_setting.pt"
        nll_file_path = os.path.join(self.args.data_dir, nll_filename)
        fednll_scene = torch.load(nll_file_path)
        self.each_noise_ratio = [fednll_scene['noise_ratio'][cid] for cid in sorted(fednll_scene['noise_ratio'].keys())]        
        self.est_noise_ratio = self.each_noise_ratio
        self.ASYM_LOSS = []
        self.SYM_LOSS = []
        self.CLEAN_LOSS = []
        self.TST_ASYM_LOSS = []
        self.TST_SYM_LOSS = []
        self.TST_CLEAN_LOSS = []
        self.TR_ASYM_LOSS = []
        self.TR_SYM_LOSS = []
        self.TR_CLEAN_LOSS = []
        self.PREDI_LABELS = [[] for c in range(args.num_clients)]
        self.MEM_ASYM_RATIO = []
        self.MEM_SYM_RATIO = []
        self.MEM_CLEAN_RATIO = []
        self.TR_CLNT_LOSS = []
        self.RECALL_DETEC = {'sym':[],'asym':[]}
        self.PRECISION_DETEC = {'sym':[],'asym':[]}
        self.RECALL_SUDO = {'sym':[],'asym':[]}
        self.PRECISION_SUDO = {'sym':[],'asym':[]}
        self.sudo_labels = [[] for i in range(args.num_clients)]
        
        self.w_start = 0

    @property
    def model_parameters(self) -> torch.Tensor:
        return misc.serialize_model(self._model)

    def set_model(self, parameters: torch.Tensor):
        misc.deserialize_model(self._model, parameters)

    def setup_optim(self, epochs, batch_size, lr, weight_decay, momentum):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.SGD(
            self._model.parameters(), lr, weight_decay=weight_decay, momentum=momentum
        )
        self.criterion = get_robust_loss(CLASS_NUM[self.args.dataset], self.args)

    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        self.cache = []
        return package

    def class_num(self,cid):
        data_loader = self.dataset.get_dataloader(
            cid=cid, train=True, batch_size=self.batch_size
        )
        havecls_num = np.bincount(data_loader.dataset.noisy_labels)
        class_nm = np.zeros(CLASS_NUM[self.args.dataset])
        for c in range(len(havecls_num)):
            class_nm[c] = havecls_num[c]
        return class_nm
    def most_common_in_columns(self,matrix):
        most_common = []
        matrix = torch.stack(matrix)
        for column in matrix.transpose(0,1):  # Transpose matrix to iterate over columns
            unique, counts = torch.unique(column, return_counts=True)
            most_common_value = unique[counts.argmax()]
            most_common.append(most_common_value.item())
        return most_common

    def mem_noise(self,prediction_clip,noise_mask,noisy_labels):
        noise_num = noise_mask.sum().item()
        datasize = len(noise_mask)
        clean_num = datasize-noise_num
        most_common_predict = self.most_common_in_columns(prediction_clip)
        memorize = torch.tensor(most_common_predict).eq(torch.tensor(noisy_labels)).to(noise_mask.device)
        memorize_noise = memorize[torch.where(noise_mask)].sum().item()
        memorize_clean = memorize.sum().item() - memorize_noise
        self.mem_num[self.clnt_mode[self.cur_cid]] += memorize_noise
        self.mem_num['clean'] += memorize_clean
        self.num[self.clnt_mode[self.cur_cid]] += noise_num
        self.num['clean'] += clean_num

    def append_rnd_conf_metric(self,):
        for mode in self.RECALL_SUDO.keys():
            self.RECALL_SUDO[mode].append(np.mean(self.recall_sudo_all[mode]))
            self.PRECISION_SUDO[mode].append(np.mean(self.precision_sudo_all[mode]))
            self.RECALL_DETEC[mode].append(np.mean(self.recall_det_all[mode]))
            self.PRECISION_DETEC[mode].append(np.mean(self.precision_det_all[mode]))
        self.watcher.plot_conf_metric(self.RECALL_SUDO, self.PRECISION_SUDO, self.cmp_out_dir, aspect='sudo')
        self.watcher.plot_conf_metric(self.RECALL_DETEC, self.PRECISION_DETEC, self.cmp_out_dir, aspect='detec')
        with open(os.path.join(self.cmp_out_dir, 'conf_metric.pkl'), 'wb') as f:
            pickle.dump({'RECALL_SUDO': self.RECALL_SUDO, 'PRECISION_SUDO': self.PRECISION_SUDO,
                         'RECALL_DETEC': self.RECALL_DETEC, 'PRECISION_DETEC': self.PRECISION_DETEC}
                        , f)

    def append_clnt_conf_metric(self,payload,):
        confidence, outputs = self.class_confidence(self.cur_cid, payload)
        if self.round == self.args.sudo:
            self.confidence_each.append(confidence)
        if self.args.sudo_once:
            if self.round <= self.args.sudo:
                self.sudo_labels[self.cur_cid] = self.sudo_labeling(self.cur_cid, confidence, outputs)
        else:
            self.sudo_labels[self.cur_cid] = self.sudo_labeling(self.cur_cid, confidence,outputs)
        detect_mask, recall_det, precision_det, recall_sudo, precision_sudo = self.detect_noise(self.cur_cid, self.sudo_labels[self.cur_cid])
        if self.clnt_mode[self.cur_cid] != 'clean':
            self.recall_sudo_all[self.clnt_mode[self.cur_cid]].append(recall_sudo)
            self.precision_sudo_all[self.clnt_mode[self.cur_cid]].append(precision_sudo)
            self.recall_det_all[self.clnt_mode[self.cur_cid]].append(recall_det)
            self.precision_det_all[self.clnt_mode[self.cur_cid]].append(precision_det)


    def watch_metric(self,payload=None, data_loader=None,func=None):
        if self.args.watchtrloss:
            with open(os.path.join(self.cmp_out_dir, 'tr_loss.pkl'), 'rb') as f:
                self.TR_CLNT_LOSS = pickle.load(f)
            self.watcher.plot_trclntLoss(self.TR_CLNT_LOSS, self.clnt_mode,
             os.path.join(self.cmp_out_dir, 'train_loss_new.png'))
            sys.exit()
        model_parameters = payload[0]
        if self.args.watch_conf:
            self.append_clnt_conf_metric(payload)
        if self.args.glob_memory:
            self.global_memory(model_parameters, data_loader)
        if self.args.memory:
            trloss_noise, trloss_clean, trloss_sum, noise_num = func(data_loader)
            self.tr_loss[self.clnt_mode[self.cur_cid]].append(trloss_sum.item() / len(data_loader.dataset))
            self._loss[self.clnt_mode[self.cur_cid]] += trloss_noise
            self._loss['clean'] += trloss_clean
            self._num[self.clnt_mode[self.cur_cid]] += noise_num
            self._num['clean'] += len(data_loader.dataset) - noise_num

    def load_mem(self):
        path = os.path.join(self.cmp_out_dir, 'conf_metric.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                load_dict = pickle.load(f)
                self.RECALL_SUDO = load_dict['RECALL_SUDO']
                self.PRECISION_SUDO = load_dict['PRECISION_SUDO']
                self.RECALL_DETEC = load_dict['RECALL_DETEC']
                self.PRECISION_DETEC = load_dict['PRECISION_DETEC']
        path = os.path.join(self.cmp_out_dir, 'trLoss_examp.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                load_dict = pickle.load(f)
                self.SYM_LOSS = load_dict['SYM_LOSS']
                self.ASYM_LOSS = load_dict['ASYM_LOSS']
                self.CLEAN_LOSS = load_dict['CLEAN_LOSS']
        path = os.path.join(self.cmp_out_dir, 'trLoss.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                load_dict = pickle.load(f)
                self.TR_SYM_LOSS = load_dict['SYM_LOSS']
                self.TR_ASYM_LOSS = load_dict['ASYM_LOSS']
                self.TR_CLEAN_LOSS = load_dict['CLEAN_LOSS']
        path = os.path.join(self.cmp_out_dir, 'tstLoss.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                load_dict = pickle.load(f)
                self.TST_SYM_LOSS = load_dict['SYM_LOSS']
                self.TST_ASYM_LOSS = load_dict['ASYM_LOSS']
                self.TST_CLEAN_LOSS = load_dict['CLEAN_LOSS']
        path = os.path.join(self.cmp_out_dir, 'memRatio.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                load_dict = pickle.load(f)
                self.MEM_SYM_RATIO = load_dict['SYM_'][:self.args.sudo-self.args.window-1]
                self.MEM_ASYM_RATIO = load_dict['ASYM_'][:self.args.sudo-self.args.window-1]
                self.MEM_CLEAN_RATIO = load_dict['CLEAN_'][:self.args.sudo-self.args.window-1]
        path = os.path.join(self.cmp_out_dir, 'pred_labels.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.PREDI_LABELS = pickle.load(f)
        debug = True

    def plot_memory(self):

        if self.args.memory:
            self.watcher.plot_trLoss_examp(self.SYM_LOSS,self.ASYM_LOSS,self.CLEAN_LOSS,self.cmp_out_dir)
        if self.args.watchtst:
            self.watcher.plot_tstLoss(self.TST_SYM_LOSS,self.TST_ASYM_LOSS,self.TST_CLEAN_LOSS,self.cmp_out_dir)
        if self.args.watchtr:
            self.watcher.plot_trLoss(self.TR_SYM_LOSS,self.TR_ASYM_LOSS,self.TR_CLEAN_LOSS,self.cmp_out_dir)
        if self.args.glob_memory and self.round >= self.args.window:
            self.watcher.plot_memRatio(self.MEM_CLEAN_RATIO, self.MEM_SYM_RATIO,
                                       os.path.join(self.cmp_out_dir, 'memRatio_sym.png'), start=self.args.window)
            self.watcher.plot_memRatio(self.MEM_CLEAN_RATIO, self.MEM_ASYM_RATIO,
                                       os.path.join(self.cmp_out_dir, 'memRatio_asym.png'), start=self.args.window)


    def save_memory(self,):
        if self.round == self.args.sudo:
            with open(os.path.join(self.cmp_out_dir,'sudo_confidence.pkl'),'wb') as f:
                pickle.dump(self.confidence_each,f)
        if self.args.watch_conf:
            self.append_rnd_conf_metric()
        if self.args.memory:
            avg_asym_loss = self._loss['asym']/self._num['asym']
            avg_sym_loss = self._loss['sym']/self._num['sym']
            avg_clean_loss = self._loss['clean']/self._num['clean']
            self.SYM_LOSS.append(avg_sym_loss.item())
            self.ASYM_LOSS.append(avg_asym_loss.item())
            self.CLEAN_LOSS.append(avg_clean_loss.item())
            self.watcher.plot_trLoss_examp(self.SYM_LOSS,self.ASYM_LOSS,self.CLEAN_LOSS,self.cmp_out_dir)
            with open(os.path.join(self.cmp_out_dir,'trLoss_examp.pkl'),'wb') as f:
                pickle.dump({'SYM_LOSS':self.SYM_LOSS,'ASYM_LOSS':self.ASYM_LOSS,'CLEAN_LOSS':self.CLEAN_LOSS}
                ,f)
        if self.args.watchtst:
            self.TST_SYM_LOSS.append(np.mean(self.tst_loss['sym']))
            self.TST_ASYM_LOSS.append(np.mean(self.tst_loss['asym']))
            self.TST_CLEAN_LOSS.append(np.mean(self.tst_loss['clean']))
            self.watcher.plot_tstLoss(self.TST_SYM_LOSS,self.TST_ASYM_LOSS,self.TST_CLEAN_LOSS,self.cmp_out_dir)
            with open(os.path.join(self.cmp_out_dir,'tstLoss.pkl'),'wb') as f:
                pickle.dump({'SYM_LOSS':self.TST_SYM_LOSS,'ASYM_LOSS':self.TST_ASYM_LOSS,'CLEAN_LOSS':self.TST_CLEAN_LOSS}
                ,f)
        if self.args.watchtr:
            self.TR_SYM_LOSS.append(np.mean(self.tr_loss['sym']))
            self.TR_ASYM_LOSS.append(np.mean(self.tr_loss['asym']))
            self.TR_CLEAN_LOSS.append(np.mean(self.tr_loss['clean']))
            self.watcher.plot_trLoss(self.TR_SYM_LOSS,self.TR_ASYM_LOSS,self.TR_CLEAN_LOSS,self.cmp_out_dir)
            with open(os.path.join(self.cmp_out_dir,'trLoss.pkl'),'wb') as f:
                pickle.dump({'SYM_LOSS':self.TR_SYM_LOSS,'ASYM_LOSS':self.TR_ASYM_LOSS,'CLEAN_LOSS':self.TR_CLEAN_LOSS}
                ,f)
        if self.args.glob_memory and self.round >= self.args.window:
                mem_sym_ra = self.mem_num['sym'] / self.num['sym'] if self.num['sym'] != 0 else 0
                mem_asym_ra = self.mem_num['asym'] / self.num['asym'] if self.num['asym'] != 0 else 0
                mem_clean_ra = self.mem_num['clean'] / self.num['clean'] if self.num['clean'] != 0 else 0
                self.MEM_SYM_RATIO.append(mem_sym_ra)
                self.MEM_ASYM_RATIO.append(mem_asym_ra)
                self.MEM_CLEAN_RATIO.append(mem_clean_ra)
                self.watcher.plot_memRatio(self.MEM_CLEAN_RATIO, self.MEM_SYM_RATIO,
                                           os.path.join(self.cmp_out_dir, 'memRatio_sym.png'), start=self.args.window)
                self.watcher.plot_memRatio(self.MEM_CLEAN_RATIO, self.MEM_ASYM_RATIO,
                                           os.path.join(self.cmp_out_dir, 'memRatio_asym.png'), start=self.args.window)
                with open(os.path.join(self.cmp_out_dir, 'pred_labels.pkl'), 'wb') as f:
                    pickle.dump(self.PREDI_LABELS, f)
                with open(os.path.join(self.cmp_out_dir, 'memRatio.pkl'), 'wb') as f:
                    pickle.dump(
                        {'SYM_': self.MEM_SYM_RATIO, 'ASYM_': self.MEM_ASYM_RATIO, 'CLEAN_': self.MEM_CLEAN_RATIO}
                        , f)

    def init_memory(self):
        self._loss = {'sym': 0, 'asym': 0, 'clean': 0}
        self._num = {'sym': 0, 'asym': 0, 'clean': 0}
        self.mem_num = {'sym': 0, 'asym': 0, 'clean': 0}
        self.num = {'sym': 0, 'asym': 0, 'clean': 0}
        self.tst_loss = {'sym': [], 'asym': [], 'clean': []}
        self.tr_loss = {'sym': [], 'asym': [], 'clean': []}
        self.recall_det_all = {'sym': [], 'asym': []}
        self.recall_sudo_all = {'sym': [], 'asym': []}
        self.precision_det_all = {'sym': [], 'asym': []}
        self.precision_sudo_all = {'sym': [], 'asym': []}
        self.confidence_each = []


    def local_process(self, payload, id_list, cur_round):
        self.round = cur_round
        self.logger.debug(f"Round {self.round} selected clients: {id_list}")
        model_parameters = payload[0]
        self.init_memory()
        #=======load_memory=================!!!
        if self.args.load_last:
            self.load_mem()
        # self.plot_memory()
        for cid in range(self.args.num_clients):
            self.cur_cid = cid
            data_loader = self.dataset.get_dataloader(
                cid=cid, train=True, batch_size=self.batch_size
            )
            pack = self.train(model_parameters, data_loader)

            self.cache.append(pack)
            loss_, acc_ = self.evaluate()
            self.logger.debug(
                f"Round {self.round} client-{self.cur_cid} local test accuracy: {acc_*100:.2f}%, local test loss: {loss_:.4f}"
            )
        # self.TR_CLNT_LOSS.append(loss_rnd)
        # with open(os.path.join(self.cmp_out_dir, 'tr_loss.pkl'),'wb') as f:
        #     pickle.dump(self.TR_CLNT_LOSS,f)
        # self.watcher.plot_trclntLoss(self.TR_CLNT_LOSS,self.clnt_mode,os.path.join(self.cmp_out_dir, 'train_loss.png'))
        # self.save_memory()
            

    def class_confidence(self,cid, payload,):
        model_parameters = payload[0]
        self.set_model(model_parameters)
        loader = self.dataset.get_dataloader(
            cid=cid, train=True, batch_size=self.batch_size, shuffle=False
        )
        data_size = len(loader.dataset)
        self._model.eval() # 一般是用全局模型
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self_confidence = torch.zeros(data_size).to(self.device)
        outputs_ = torch.zeros(data_size,CLASS_NUM[self.args.dataset]).to(self.device)
        with torch.no_grad():
            for batch_idx, (imgs, labels, noisy_labels, index) in enumerate(loader):
                imgs = imgs.cuda(self.device)
                noisy_labels = noisy_labels.cuda(self.device)
                outputs = self.model(imgs)
                sft_outputs = torch.softmax(outputs, dim=1)
                outputs_[index] = sft_outputs
                # # NEW: mask out the noisy label position and find the max over others
                # one_hot = F.one_hot(noisy_labels, CLASS_NUM[self.args.dataset]).bool()
                # sft_outputs_masked = sft_outputs.masked_fill(one_hot, float('-inf'))  # exclude noisy label prob
                # alt_confidence = torch.max(sft_outputs_masked, dim=1)[0]  # max prob not on noisy label
                # self_confidence[index] = 1-alt_confidence
                one_hot_lab = F.one_hot(noisy_labels,CLASS_NUM[self.args.dataset]).float()
                self_confidence[index] = torch.sum(sft_outputs*one_hot_lab,axis=1)


        confidence_cl = np.zeros(CLASS_NUM[self.args.dataset])
        for idx in range(len(loader.dataset)):
            c = loader.dataset.noisy_labels[idx]
            confidence_cl[c] += self_confidence[idx].item()
        confidence_cl = confidence_cl / self.class_num(cid)
        confidence_cl = np.nan_to_num(confidence_cl)
        return confidence_cl,outputs_

    def class_loss(self, cid, payload, softmax=False):
        criterion = nn.CrossEntropyLoss(reduction='none')

        model_parameters = payload[0]
        self.set_model(model_parameters)
        loader = self.dataset.get_dataloader(
            cid=cid, train=True, batch_size=self.batch_size,shuffle=False
        )

        self._model.eval()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with torch.no_grad():
            for i, (images, _, labels, index) in enumerate(loader):
                images = images.to(self.device)

                # Forward pass
                logits = self._model(images)
                probs = F.softmax(logits, dim=1)

                # Compute KL divergence to uniform distribution
                # Uniform target: [1/C, 1/C, ..., 1/C]
                kl_div = torch.sum(probs * torch.log(probs * CLASS_NUM + 1e-8), dim=1)

                if i == 0:
                    output_whole = probs.cpu().numpy()
                    loss_whole = kl_div.cpu().numpy()
                else:
                    output_whole = np.concatenate((output_whole, probs.cpu().numpy()), axis=0)
                    loss_whole = np.concatenate((loss_whole, kl_div.cpu().numpy()), axis=0)

        # Aggregate by noisy label class
        metric_id = np.zeros(CLASS_NUM)
        for idx in range(len(loader.dataset)):
            c = loader.dataset.noisy_labels[idx]
            metric_id[c] += loss_whole[idx]
        # with torch.no_grad():
        #     for i, (images,_,labels,index) in enumerate(loader):
        #         images = images.to(self.device)
        #         labels = labels.to(self.device)
        #         labels = labels.long()
        #         if softmax == True:
        #             outputs = self._model(images)
        #             outputs = F.softmax(outputs, dim=1)
        #         else:
        #             outputs = self._model(images)
        #         if criterion is not None:
        #             loss = criterion(outputs, labels)
        #         if i == 0:
        #             output_whole = np.array(outputs.cpu())
        #             if criterion is not None:
        #                 loss_whole = np.array(loss.cpu())
        #         else:
        #             output_whole = np.concatenate(
        #                 (output_whole, outputs.cpu()), axis=0)
        #             if criterion is not None:
        #                 loss_whole = np.concatenate(
        #                     (loss_whole, loss.cpu()), axis=0)

        # metric_id = np.zeros(CLASS_NUM[self.args.dataset])
        # for idx in range(len(loader.dataset)):
        #     c = loader.dataset.noisy_labels[idx]
        #     metric_id[c] += loss_whole[idx]
        metric_id  = metric_id/self.class_num(cid)
        metric_id = np.nan_to_num(metric_id)
        return metric_id

    def sudo_labeling(self,cid,confidence_cl,outputs_):
        loader = self.dataset.get_dataloader(
            cid=cid, train=True, batch_size=self.batch_size, shuffle=False
        )
        confidence_cl = torch.tensor(np.repeat(confidence_cl.reshape(1,CLASS_NUM[self.args.dataset]),len(outputs_),axis=0)).to(self.device)
        conf_filter = (outputs_ > confidence_cl).float()
        conf_outputs = outputs_* conf_filter
        sudo_labels = torch.argmax(conf_outputs,dim=1)

        return sudo_labels

    def detect_noise(self,cid,sudo_labels):
        loader = self.dataset.get_dataloader(
            cid=cid, train=True, batch_size=self.batch_size, shuffle=False
        )
        noisy_labels = torch.tensor(loader.dataset.noisy_labels).to(self.device)
        true_labels = torch.tensor(loader.dataset.labels).to(self.device)
        sudo_mask = (noisy_labels).eq(sudo_labels)
        noise_mask = (noisy_labels).eq(true_labels)
        true_positive = (sudo_mask * noise_mask).sum().item()#判定为噪声且确实为噪声
        noise_num = noise_mask.sum().item()
        detec_recall = true_positive / noise_num if noise_num > 0 else 0
        positive_num = sudo_mask.sum().item()
        detec_precision = true_positive / positive_num if positive_num > 0 else 0


        true_positive = (sudo_mask * (sudo_labels.eq(true_labels))).sum().item()  # 判定为噪声且改对了
        sudo_recall = true_positive / noise_num if noise_num > 0 else 0
        positive_num = sudo_mask.sum().item()
        sudo_precision = true_positive / positive_num if positive_num > 0 else 0

        return sudo_mask,detec_recall,detec_precision,sudo_recall,sudo_precision

    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self.setup_optim(
            self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum
        )
        self._model.train()
        data_size = len(train_loader.dataset)

        for epoch in range(self.epochs):

            batch_num = len(train_loader)
            for batch_idx,(imgs, labels, noisy_labels,index) in enumerate(train_loader):
                if self.cuda:
                    imgs = imgs.cuda(self.device)
                    noisy_labels = noisy_labels.cuda(self.device)

                outputs = self.model(imgs)
                loss = self.criterion(outputs, noisy_labels)

                self.optimizer.zero_grad()
                self._model.zero_grad()
                loss.backward()
                self.optimizer.step()
                sys.stdout.write('\r')
                sys.stdout.write('User = %d  | Global Epoch %d | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f'
                                 % (self.cur_cid, self.round, epoch, self.epochs, batch_idx + 1, batch_num,
                                    loss.item()))
                sys.stdout.flush()

        local_result = [self.model_parameters, data_size]
        return local_result

    def count_noise(self,train_loader):
        data_size = len(train_loader.dataset)
        self.clean_bool = np.zeros(data_size)
        self.sym_bool = np.zeros(data_size)
        self.asym_bool = np.zeros(data_size)
        for batch_idx, (imgs, labels, noisy_labels, index) in enumerate(train_loader):
            labels = labels.cuda(self.device)
            noisy_labels = noisy_labels.cuda(self.device)

            self.clean_bool[index] = [1 if element else 0 for element in labels.eq(noisy_labels)]
            for idx in index:
                if self.clean_bool[idx] == 0:
                    if idx in self.sym_idxs:
                        self.sym_bool[idx] = 1
                    else:
                        self.asym_bool[idx] = 1
        noise_rate = 1-self.clean_bool.sum()/data_size
        sym_rate = self.sym_bool.sum()/data_size
        asym_rate = self.asym_bool.sum()/data_size
        sys.stdout.write('\r')
        sys.stdout.write('this dataset has {:.2f}% label noise, {:.2f}% asymmetric and {:.2f}% symmetric'.format(noise_rate*100,sym_rate*100,asym_rate*100))
        sys.stdout.flush()

    def local_loss(self,model_parameters,train_loader):
        self.set_model(model_parameters)
        self._model.eval()
        data_size = len(train_loader.dataset)
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss_sum = 0
        with torch.no_grad():
            for batch_idx, (imgs, labels, noisy_labels, index) in enumerate(train_loader):
                imgs = imgs.cuda(self.device)
                labels = labels.cuda(self.device)
                noisy_labels = noisy_labels.cuda(self.device)
                outputs = self.model(imgs)
                loss = criterion(outputs,noisy_labels)
                loss_sum += torch.sum(loss)
        average_loss = loss_sum/data_size
        average_loss = average_loss.item()
        return average_loss

    def global_memory(self,model_parameters,train_loader):
        self.set_model(model_parameters)
        self._model.eval()
        data_size = len(train_loader.dataset)
        pred_label = torch.zeros(data_size).long().cuda(self.device)
        noise_mask = torch.zeros(data_size).long().cuda(self.device)
        with torch.no_grad():
            for batch_idx, (imgs, labels, noisy_labels, index) in enumerate(train_loader):
                imgs = imgs.cuda(self.device)
                labels = labels.cuda(self.device)
                noisy_labels = noisy_labels.cuda(self.device)
                outputs = self.model(imgs)
                _, predicted = torch.max(outputs, 1)
                pred_label[index] = predicted
                noise_mask[index] = (labels != noisy_labels).long()
        noise_num = noise_mask.sum()

        if self.round >= 1:
            self.PREDI_LABELS[self.cur_cid].append(pred_label)
        if self.round >= self.args.window:  # 训练轮数大于窗口轮数后开始计算本地记忆数和本地数据量
            # 一个滑窗的记忆历史
            prediction_clip = self.PREDI_LABELS[self.cur_cid][self.round - self.args.window:self.round]
            self.mem_noise(prediction_clip, noise_mask, train_loader.dataset.noisy_labels)

    def eval_memory(self, train_loader):
        self._model.eval()
        data_size = len(train_loader.dataset)
        outputs_rst = torch.empty(data_size,CLASS_NUM[self.args.dataset]).cuda(self.device)
        pred_label = torch.zeros(data_size).long().cuda(self.device)
        noise_mask = torch.zeros(data_size).long().cuda(self.device)
        loss_whole = torch.zeros(data_size).cuda(self.device)
        criterion = get_robust_loss(CLASS_NUM[self.args.dataset], self.args,reduction='none')
        with torch.no_grad():
            for batch_idx, (imgs, labels, noisy_labels,index) in enumerate(train_loader):
                imgs = imgs.cuda(self.device)
                labels = labels.cuda(self.device)
                noisy_labels = noisy_labels.cuda(self.device)

                outputs = self.model(imgs)
                _, predicted = torch.max(outputs, 1)
                pred_label[index] = predicted
                # pred_rst[index] = [1 if element else 0 for element in predicted.eq(noisy_labels)]
                noise_mask[index] = (labels != noisy_labels).long()
                outputs_rst[index] = outputs
                loss = criterion(outputs, noisy_labels)
                loss_whole[index] = loss
        loss_noise = loss_whole[torch.where(noise_mask)].sum()
        loss_clean = loss_whole.sum()-loss_noise
        noise_num = noise_mask.sum()
        # self.watcher.memorize_ratio(pred_label,train_loader.dataset.noisy_labels,noise_mask)
        # if self.round % 5 ==0:
        #     self.watcher.plot_memoryHis(self.clnt_mode[self.cur_cid],self.each_noise_ratio[self.cur_cid],self.cmp_out_dir)

        return loss_noise,loss_clean,loss_whole.sum(),noise_num

    def train_watchmix(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self.setup_optim(
            self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum
        )
        self._model.train()
        data_size = len(train_loader.dataset)
        pred_rst = np.zeros(data_size) # the number of forgetting events during the following epochs
        outputs_rst = torch.empty(data_size,CLASS_NUM[self.args.dataset]).cuda(self.device)
        pred_label = torch.zeros((data_size,self.epochs)).long().cuda(self.device)
        noise_mask = torch.zeros(data_size).cuda(self.device)
        for epoch in range(self.epochs):
            batch_num = len(train_loader)
            for batch_idx, (imgs, labels, noisy_labels,index) in enumerate(train_loader):
                imgs = imgs.cuda(self.device)
                labels = labels.cuda(self.device)
                noisy_labels = noisy_labels.cuda(self.device)

                outputs = self.model(imgs)
                _, predicted = torch.max(outputs, 1)
                pred_label[index,epoch] = predicted
                # pred_rst[index] = [1 if element else 0 for element in predicted.eq(noisy_labels)]
                noise_mask[index] = (labels != noisy_labels).float()
                outputs_rst[index] = outputs
                loss = self.criterion(outputs, noisy_labels)

                self.optimizer.zero_grad()
                self._model.zero_grad()
                loss.backward()
                self.optimizer.step()
                sys.stdout.write('\r')
                sys.stdout.write('User = %d  | Global Epoch %d | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f'
                                 % (self.cur_cid, self.round, epoch, self.epochs, batch_idx + 1, batch_num,
                                    loss.item()))
                sys.stdout.flush()

            # self.watcher.update_event(pred_result=pred_rst)

        #     if epoch == 0:
        #         self.watcher.update_shift(0, outputs_rst, train_loader.dataset.noisy_labels)
        # self.watcher.update_shift(-1,outputs_rst,train_loader.dataset.noisy_labels)
        # self.watcher.curr_eventNm(self.round, self.clean_bool, self.sym_bool, self.asym_bool,
        #                                savepath=self.cmp_out_dir)
        self.watcher.memorize_ratio(pred_label,train_loader.dataset.noisy_labels,noise_mask)
        if self.round % 5 ==0:
            self.watcher.plot_memoryHis(self.clnt_mode[self.cur_cid],self.each_noise_ratio[self.cur_cid],self.cmp_out_dir)
            # self.watcher.plot_eventHis(self.round,self.clean_bool,self.sym_bool,self.asym_bool,savepath=self.cmp_out_dir)
            # self.watcher.plot_shiftHis(self.round,self.clean_bool,self.sym_bool,self.asym_bool,savepath=self.cmp_out_dir)
        local_result = [self.model_parameters, data_size]
        return local_result



    def evaluate(self):
        test_loader = self.dataset.get_dataloader(train=False, batch_size=128)
        multimodel = hasattr(self._model, "models")
        loss_, acc_ = misc.evaluate(
            self._model,
            nn.CrossEntropyLoss(),
            test_loader,
            self.device,
            multimodel=multimodel,
        )

        return loss_, acc_


class FedNLLFedAvgESClientTrainer(FedNLLFedAvgClientTrainer):
    def __init__(
            self,
            model,
            num_clients,
            cuda=True, device=None,
            loggerfile=None,
            personal=False,
            args=None,
    ) -> None:
        FedNLLFedAvgClientTrainer.__init__(
            self,
            model,
            num_clients,
            cuda,
            device,
            loggerfile,
            personal,
            args,
        )
        self.cache = []
        self.args = args
        writer = SummaryWriter(comment='scalar')
        self.logger.debug(writer.log_dir.replace('runs/', ''))
        self.watcher = WatchSystematicNoise(self.args.num_clients)
        nll_name = nllF.FedNLL_name(**vars(args))
        exp_name = make_exp_name("fedavg", args)
        alg_name = make_alg_name(args)
        #=================noise setting directory=========================
        self.nll_name = nllF.FedNLL_name(**vars(args))
        nll_filename = f"{self.nll_name}_seed_{args.seed}_setting.pt"
        nll_file_path = os.path.join(self.args.data_dir, nll_filename)
        fednll_scene = torch.load(nll_file_path)
        self.cmp_out_dir = os.path.join(args.out_dir, nll_name, alg_name, exp_name)
        if self.args.noise_mode == 'uniform':
            self.clnt_mode = fednll_scene['clnt_mode']
        self.sudo_labels = [[] for i in range(args.num_clients)]


    def setup_optim(self, epochs, batch_size, lr, weight_decay, momentum,alpha=0.1):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.SGD(
            self._model.parameters(), lr, weight_decay=weight_decay, momentum=momentum
        )
        self.criterion = get_robust_loss(CLASS_NUM[self.args.dataset], self.args,alpha)

    
    def count_key_appearances(self,dictionary):
        counts = {'sym':0,'asym':0,'clean':0}
        for mode in dictionary.values():
            counts[mode] += 1
        return counts
    @property
    def model_parameters(self) -> torch.Tensor:
        return misc.serialize_model(self._model)

    def set_model(self, parameters: torch.Tensor):
        misc.deserialize_model(self._model, parameters)

    def setup_optim(self, epochs, batch_size, lr, weight_decay, momentum,alpha=0.1):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.SGD(
            self._model.parameters(), lr, weight_decay=weight_decay, momentum=momentum
        )
        self.criterion = get_robust_loss(CLASS_NUM[self.args.dataset], self.args,alpha)

    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        self.cache = []
        return package

    def cal_metric(self):
        to_concat_g = []
        for key in self.old_model_state.keys():
            if self.old_model_state[key].dim() in [2, 4]:
                grad = (self._model.state_dict()[key] - self.old_model_state[key]) / self.lr
                to_concat_g.append(grad.view(-1))
        all_g = torch.cat(to_concat_g)
        self.metric = torch.abs(all_g).cpu().numpy()


    def compute_trust_score(self, payload, cid):
        ce_total = 0.0
        confs = []
        criterion = nn.CrossEntropyLos(reduction='sum')
        model_parameters = payload[0]
        self.set_model(model_parameters)
        self._model.eval()
        total_samples = 0
        loader = self.dataset.get_dataloader(
            cid=cid, train=True, batch_size=self.batch_size, shuffle=False
        )
        with torch.no_grad():
            for batch_idx, (imgs, labels, noisy_labels, index) in enumerate(loader):
                imgs, noisy_labels = imgs.to(self.device), noisy_labels.to(self.device)
                logits = self._model(imgs)

                # 交叉熵计算
                ce = criterion(logits, noisy_labels)
                ce_total += ce.item()
                total_samples += noisy_labels.size(0)

                # 置信度为 softmax 最大值
                probs = F.softmax(logits, dim=1)
                max_probs = probs.max(dim=1)[0]
                confs.extend(max_probs.cpu().numpy())

        # 计算平均交叉熵并归一化到 [0, 1]
        avg_ce = ce_total / total_samples
        norm_ce = min(avg_ce / 5.0, 1.0)  # 假设交叉熵最大值不超过 5

        # 计算置信度方差并归一化（0 表示稳定高置信；1 表示波动大）
        var_conf = torch.tensor(confs).var().item()
        norm_var = min(var_conf / 0.1, 1.0)  # 经验上最大方差约为 0.1

        # 组合信任因子（越接近 1 表示越可信）
        trust_score = (1 - norm_ce) * (1 - norm_var)
        return trust_score

    def class_metrics(self, cid, payload, conf_threshold=0.7):
        
        model_parameters = payload[0]
        self.set_model(model_parameters)
        loader = self.dataset.get_dataloader(
            cid=cid, train=True, batch_size=self.batch_size, shuffle=False
        )
        data_size = len(loader.dataset)
        class_num = CLASS_NUM[self.args.dataset]

        self._model.eval()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        confidence_whole = np.zeros(data_size)
        with torch.no_grad():
            for batch_idx, (imgs, labels, noisy_labels, index) in enumerate(loader):
                imgs = imgs.to(self.device)
                logits = self._model(imgs)
                probs = F.softmax(logits, dim=1)
                max_conf, _ = probs.max(dim=1)
                confidence_whole[index] = max_conf.cpu().numpy()

        # 估计噪声率：置信度低于阈值的样本比例
        est_noise_rate = np.mean(confidence_whole < conf_threshold)

        # 按类别聚合平均置信度
        metric_conf = np.zeros(class_num)
        class_counts = self.class_num(cid)
        for idx in range(data_size):
            c = loader.dataset.noisy_labels[idx]
            metric_conf[c] += confidence_whole[idx]
        for c in range(class_num):
            n = class_counts[c]
            if n > 0:
                metric_conf[c] /= n
        metric_conf = np.nan_to_num(metric_conf)

        return metric_conf, est_noise_rate


    def class_num(self,cid):
        data_loader = self.dataset.get_dataloader(
            cid=cid, train=True, batch_size=self.batch_size
        )
        havecls_num = np.bincount(data_loader.dataset.noisy_labels)
        class_nm = np.zeros(CLASS_NUM[self.args.dataset])
        for c in range(len(havecls_num)):
            class_nm[c] = havecls_num[c]
        return class_nm

    def class_feature(self, cid, payload):
        criterion = nn.CrossEntropyLoss(reduction='none')
        model_parameters = payload[0]
        self.set_model(model_parameters)
        loader = self.dataset.get_dataloader(
            cid=cid, train=True, batch_size=self.batch_size, shuffle=False
        )

        self._model.eval()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        metric_id = np.zeros(CLASS_NUM[self.args.dataset])
        feature_whole = torch.zeros((len(loader.dataset),))
        with torch.no_grad():
            for i, (images, labels, _, index) in enumerate(loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                labels = labels.long()
                outputs = self._model(images,latent_output=True)
                if criterion is not None:
                    loss = criterion(outputs, labels)
                if i == 0:
                    output_whole = np.array(outputs.cpu())
                    if criterion is not None:
                        loss_whole = np.array(loss.cpu())
                else:
                    output_whole = np.concatenate(
                        (output_whole, outputs.cpu()), axis=0)
                    if criterion is not None:
                        loss_whole = np.concatenate(
                            (loss_whole, loss.cpu()), axis=0)
        for idx in range(len(loader.dataset)):
            c = loader.dataset.noisy_labels[idx]
            metric_id[c] += loss_whole[idx]
        return metric_id



    def sigmoid_rampup(self,current, begin, end):
        #the return result is always larger than 0, and first from 0.* to 1 (when current=end) and then 1 to 0.*
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        current = np.clip(current, begin, end)
        phase = 1.0 - (current - begin) / (end - begin)
        return float(np.exp(-5.0 * phase * phase))

    def compute_entropy_and_consistency(self, dataloader):
    # """
    # 对本地所有样本进行推理，输出预测一致性 & 预测熵的客户端平均值
    # """
        prediction_history = defaultdict(list)
        entropies = []

        self._model.eval()
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                logits = self._model(x)
                probs = F.softmax(logits, dim=1)
                pred = probs.argmax(dim=1).cpu().numpy()
                for i, p in enumerate(pred):
                    prediction_history[i].append(p)
                entropies.extend([entropy(p.cpu().numpy()) for p in probs])

        # 计算一致性
        consistent_ratios = []
        for preds in prediction_history.values():
            if len(preds) == 0:
                continue
            dominant = max(set(preds), key=preds.count)
            consistent_ratios.append(preds.count(dominant) / len(preds))

        avg_consistency = sum(consistent_ratios) / len(consistent_ratios)
        avg_entropy = sum(entropies) / len(entropies)
        return avg_consistency, avg_entropy

    def local_preprocess(self, payload, id_list, cur_round):
        # print(type(payload),type(payload[0]),type(payload[1]))
        # print(len(payload[0]))
        # print(type(payload[0][0]))
        user_id = list(range(self.args.num_clients))
        self.round = cur_round
        self.logger.debug(f"Round {self.round} selected clients: {id_list}")
        model_parameters = copy.deepcopy(payload[0])
        weight_kd = self.sigmoid_rampup(
            cur_round, self.args.begin, self.args.end) * self.args.a
        for idx in id_list:  # training over the subset
            self.cur_cid = idx
            data_loader = self.dataset.get_dataloader(
                cid=idx, train=True, batch_size=self.batch_size
            )
            if self.args.warm_LA:
                pack = self.train_LA(model_parameters, data_loader)
            else:
                pack = self.train(model_parameters, data_loader)
            loss_, acc_ = self.evaluate()
            self.logger.debug(
                f"Round {self.round} client-{self.cur_cid} local test accuracy: {acc_ * 100:.2f}%, local test loss: {loss_:.4f}"
            )
            self.cache.append(pack)

    def train_LA(self, model_parameters,train_loader):
        self.set_model(model_parameters)
        self.setup_optim(
            self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum
        )
        self._model.train()
        batch_num = len(train_loader)
        data_size = len(train_loader.dataset)
        # set the optimizer

        # train and update
        epoch_loss = []
        ce_criterion = LogitAdjust(cls_num_list=self.class_num(self.cur_cid))

        for epoch in range(self.epochs):
            batch_loss = []
            for batch_idx, (imgs, labels, noisy_labels,index) in enumerate(train_loader):
                imgs, noisy_labels = imgs.to(self.device), noisy_labels.to(self.device)

                logits = self._model(imgs)
                loss = ce_criterion(logits, noisy_labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                sys.stdout.write('\r')
                sys.stdout.write('User = %d  | Global Epoch %d | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f'
                                 % (self.cur_cid, self.round, epoch, self.epochs, batch_idx + 1, batch_num,
                                    loss.item()))
                sys.stdout.flush()

            epoch_loss.append(np.array(batch_loss).mean())
            local_result = [self.model_parameters, data_size]

        return local_result

    def local_process_tsce_sudo(self, payload, id_list, cur_round):
        self.round = cur_round
        user_id = list(range(self.args.num_clients))
        self.logger.debug(f"Round {self.round} selected clients: {id_list}")
        model_parameters = copy.deepcopy(payload[0])
        self.init_memory()
        weight_kd = self.sigmoid_rampup(
            cur_round, self.args.begin, self.args.end) * self.args.a
        if self.args.load_last:
            self.load_mem()
            self.args.load_last = False
        for idx in id_list:  # training over the subset
            self.cur_cid = idx
            data_loader = self.dataset.get_dataloader(
                cid=idx, train=True, batch_size=self.batch_size
            )

            #===============local training ==========================

            if self.prclnt_mode[idx] == 'clean':
                pack = self.train_LA(model_parameters, data_loader)
            elif self.prclnt_mode[idx] == 'asym':  # 非对称噪声采用sce
                pack = self.train_sce(model_parameters, data_loader, teacher=True,weight_kd=weight_kd)
            else:
                if self.round >= self.args.sudo:
                    pack = self.train_sudo(model_parameters, data_loader, self.sudo_labels[self.cur_cid] )
                else:
                    pack = self.train_sce(model_parameters, data_loader, teacher=False)

            loss_, acc_ = self.evaluate()
            self.tst_loss[self.clnt_mode[self.cur_cid]].append(loss_)
            self.logger.debug(
                f"Round {self.round} client-{self.cur_cid} local test accuracy: {acc_ * 100:.2f}%, local test loss: {loss_:.4f}"
            )
            self.cache.append(pack)
        # self.save_memory()

    def local_process_mask(self, payload, id_list, cur_round):
            self.round = cur_round
            self.logger.debug(f"Round {self.round} selected clients: {id_list}")
            model_parameters = payload[0]
            self.init_memory()
            #=======load_memory=================!!!
            if self.args.load_last:
                self.load_mem()
            # self.plot_memory()
            for cid in range(self.args.num_clients):
                self.cur_cid = cid
                data_loader = self.dataset.get_dataloader(
                    cid=cid, train=True, batch_size=self.batch_size
                )
                pack = self.train_mask(model_parameters, data_loader)

                self.cache.append(pack)
                loss_, acc_ = self.evaluate()
                self.logger.debug(
                    f"Round {self.round} client-{self.cur_cid} local test accuracy: {acc_*100:.2f}%, local test loss: {loss_:.4f}"
                )
            # self.TR_CLNT_LOSS.append(loss_rnd)
            # with open(os.path.join(self.cmp_out_dir, 'tr_loss.pkl'),'wb') as f:
            #     pickle.dump(self.TR_CLNT_LOSS,f)
            # self.watcher.plot_trclntLoss(self.TR_CLNT_LOSS,self.clnt_mode,os.path.join(self.cmp_out_dir, 'train_loss.png'))
            # self.save_memory()
            

    # def train_mask(self, model_parameters, train_loader):
        
    #     self.set_model(model_parameters)
    #     self.setup_optim(
    #         self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum
    #     )
    
    #     # 1. 获取当前客户端噪声率
    #     noise_ratio = self.est_noise_ratio[self.cur_cid]
    #     freeze_ratio = min(max(noise_ratio, 0.0), 1.0)  # 0~1
    
    #     # 2. 收集可冻结参数（排除BN、bias、最后一层）
    #     named_params = list(self._model.named_parameters())
    #     freeze_candidates = []
    #     for name, param in named_params:
    #         if ('bn' in name.lower()) or ('bias' in name.lower()) or ('fc' in name.lower()) or ('classifier' in name.lower()):
    #             continue
    #         freeze_candidates.append((name, param))
    #     num_to_freeze = int(len(freeze_candidates) * freeze_ratio)
    
    #     # 3. 冻结前 freeze_ratio 部分参数
    #     for i in range(num_to_freeze):
    #         freeze_candidates[i][1].requires_grad = False
    
    #     self._model.train()
    #     criterion = nn.CrossEntropyLoss()
    #     data_size = len(train_loader.dataset)
    
    #     for epoch in range(self.epochs):
    #         batch_num = len(train_loader)
    #         for batch_idx, (imgs, labels, noisy_labels, index) in enumerate(train_loader):
    #             if self.cuda:
    #                 imgs = imgs.cuda(self.device)
    #                 noisy_labels = noisy_labels.cuda(self.device)
    #             outputs = self._model(imgs)
    #             loss = criterion(outputs, noisy_labels)
    #             self.optimizer.zero_grad()
    #             self._model.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
    #             sys.stdout.write('\r')
    #             sys.stdout.write('User = %d | mask | Global Epoch %d | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f'
    #                             % (self.cur_cid, self.round, epoch, self.epochs, batch_idx + 1, batch_num,
    #                                 loss.item()))
    #             sys.stdout.flush()
    
    #     # 4. 恢复所有参数可训练
    #     for name, param in self._model.named_parameters():
    #         param.requires_grad = True
    
    #     local_result = [self.model_parameters, data_size]
    #     return local_result
        
       
    def g_thresh(self):
            to_concat_g = []
            to_concat_v = []
            for name, param in self._model.named_parameters():
                if param.dim() in [2, 4]:
                    to_concat_g.append(param.grad.data.view(-1))
                    to_concat_v.append(param.data.view(-1))
            all_g = torch.cat(to_concat_g)
            all_v = torch.cat(to_concat_v)

            metric = torch.abs(all_g * all_v)
            self.inspect_para = len(metric)
            num_params = all_v.size(0)
            estimate_quality = 1 - self.est_noise_ratio[self.cur_cid]
            nz = int(estimate_quality * num_params)
            nz = 1 if nz == 0 else nz # nz cannot be 0
            top_values,_ = torch.topk(metric,nz)
            self.thresh = top_values[-1]

            # nz_reverse = num_params + 1 - nz
            # nz_reverse = nz_reverse if nz_reverse <= num_params else num_params
            # self.thresh, _ = torch.kthvalue(metric, k=nz_reverse)
            # sys.stdout.write('\r')
            # sys.stdout.write('nz_reverse={}'.format(nz_reverse))
            # sys.stdout.flush()

    def freeze_grad(self):
        estimate_quality = 1 - self.est_noise_ratio[self.cur_cid]
        activate_num = 0
        for name, param in self._model.named_parameters():
            if param.dim() in [2, 4]:
                mask = (torch.abs(param.data * param.grad.data) >= self.thresh).type(torch.cuda.FloatTensor)
                activate_num += mask.bool().sum().item()
                mask = mask * estimate_quality
                param.grad.data = mask * param.grad.data
        self.activate_ratio = activate_num/self.inspect_para

    
    def train_mask(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self.old_model_state = copy.deepcopy(self._model.state_dict())
        self.setup_optim(
            self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum
        )
        self._model.train()
        data_size = len(train_loader.dataset)
        estimate_quality = 1 - self.est_noise_ratio[self.cur_cid]

        for epoch in range(self.epochs):
            batch_num = len(train_loader)
            for batch_idx, (imgs, labels, noisy_labels,index) in enumerate(train_loader):
                if self.cuda:
                    imgs = imgs.cuda(self.device)
                    noisy_labels = noisy_labels.cuda(self.device)

                outputs = self._model(imgs)
                loss = self.criterion(outputs, noisy_labels)

                self.optimizer.zero_grad()
                self._model.zero_grad()
                loss.backward()
                #  freeze the non-parameters before update
                self.g_thresh()
                self.freeze_grad()
                

                self.optimizer.step()
                sys.stdout.write('\r')
                sys.stdout.write('User = %d  | Global Epoch %d | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f'
                                 % (self.cur_cid, self.round, epoch, self.epochs, batch_idx + 1, batch_num,
                                    loss.item()))
                sys.stdout.flush()
            print(
                "Quality: {}%,Activate mask:{}%".format(estimate_quality * 100, self.activate_ratio * 100))

        local_result = [self.model_parameters, data_size]
        return local_result


    def count_noise(self, train_loader):
        data_size = len(train_loader.dataset)
        self.clean_bool = np.zeros(data_size)
        self.sym_bool = np.zeros(data_size)
        self.asym_bool = np.zeros(data_size)
        for batch_idx, (imgs, labels, noisy_labels, index) in enumerate(train_loader):
            labels = labels.cuda(self.device)
            noisy_labels = noisy_labels.cuda(self.device)

            self.clean_bool[index] = [1 if element else 0 for element in labels.eq(noisy_labels)]
            for idx in index:
                if self.clean_bool[idx] == 0:
                    if idx in self.sym_idxs:
                        self.sym_bool[idx] = 1
                    else:
                        self.asym_bool[idx] = 1
        noise_rate = 1 - self.clean_bool.sum() / data_size
        sym_rate = self.sym_bool.sum() / data_size
        asym_rate = self.asym_bool.sum() / data_size
        sys.stdout.write('\r')
        sys.stdout.write(
            'this dataset has {:.2f}% label noise, {:.2f}% asymmetric and {:.2f}% symmetric'.format(noise_rate * 100,
                                                                                                    sym_rate * 100,
                                                                                                    asym_rate * 100))
        sys.stdout.flush()


    def evaluate(self):
        test_loader = self.dataset.get_dataloader(train=False, batch_size=self.batch_size)
        multimodel = hasattr(self._model, "models")
        loss_, acc_ = misc.evaluate(
            self._model,
            nn.CrossEntropyLoss(),
            test_loader,
            self.device,
            multimodel=multimodel,
        )

        return loss_, acc_

