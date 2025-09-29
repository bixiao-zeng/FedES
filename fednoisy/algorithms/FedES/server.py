import pickle
import sys
import argparse
import os
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from fedlab.core.server.manager import SynchronousServerManager

from fedlab.core.client.trainer import SerialClientTrainer
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.core.network import DistNetwork
from fedlab.utils import Logger, Aggregators, SerializationTool

sys.path.append(os.getcwd())
from fednoisy.data.NLLData import functional as nllF
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
    CIFAR10_TRANSITION_MATRIX,
    NORM_VALUES,
)

from fednoisy.utils.misc import AverageMeter
from fednoisy.utils import misc as misc
import copy
import scipy.stats as stats
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA,IncrementalPCA,TruncatedSVD

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
class FedAvgServerHandler(SyncServerHandler):
    def __init__(
        self,
        model: torch.nn.Module,
        global_round: int,
        sample_ratio: float,
        cuda: bool = True,
        device: str = None,
            loggerfile=None,
            args=None,
    ):
        SyncServerHandler.__init__(
            self, model, global_round, sample_ratio, cuda, device, logger
        )
        self.nll_name = nllF.FedNLL_name(**vars(args))
        nll_filename = f"{self.nll_name}_seed_{args.seed}_setting.pt"
        nll_file_path = os.path.join(args.data_dir, nll_filename)
        fednll_scene = torch.load(nll_file_path)
        self.each_noise_ratio = fednll_scene['noise_ratio']
        self.cmp_out_dir = os.path.dirname(loggerfile)
        print('Each noise ratio is :', self.each_noise_ratio)
        self.args = args
        if self.args.noise_mode in ['uniform', 'beta']:
            self.clnt_mode = fednll_scene['clnt_mode']

        writer = SummaryWriter(comment='scalar')
        logger.add(loggerfile, filter=lambda record: record['extra'].get('name') == 's')
        self.logger = logger.bind(name='s')
        self.logger.debug(writer.log_dir.replace('runs/', ''))
        self.batch_size = args.batch_size
    @property
    def model_parameters(self) -> torch.Tensor:
        return misc.serialize_model(self._model)


    def set_model(self, parameters: torch.Tensor):
        misc.deserialize_model(self._model, parameters)

    def setup_dataset(self, dataset) -> None:
        self.dataset = dataset

    def global_update(self, buffer):
        parameters_list = [elem[0] for elem in buffer]
        weights = [elem[1] for elem in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        self.set_model(serialized_parameters)
        # self._LOGGER.info(
        #     f"Round [{self.round}/{self.global_round}] server global update done."
        # )

    def plot_distance(self,distance,clean_clients, asym_clients,sym_clients):
        plt.figure(figsize=(10, 6))
        noisy_clients = asym_clients+sym_clients
        all_clients = list(range(self.args.num_clients))
        plt.scatter(all_clients, distance, c='gray', label='Clean Clients')
        plt.scatter(noisy_clients, distance[noisy_clients], c='red', label='Noisy Clients')
        plt.scatter(asym_clients, distance[asym_clients], c='blue',
                    label='Asymmetric Noisy Clients')
        plt.scatter(sym_clients, distance[sym_clients], c='green',
                    label='Symmetric Noisy Clients')

        plt.xlabel('Client Index')
        plt.ylabel('Normalized Distance')
        plt.legend()
        plt.title('Visualization of Client Distances')
        plt.tight_layout()
        plt.savefig(os.path.join(self.cmp_out_dir, 'distance.png'), bbox_inches='tight')

    def real_noise_clnt(self):
        clean_idx = []
        noisy_idx = []
        asym_idx = []
        sym_idx = []
        for i in range(self.args.num_clients):
            if self.clnt_mode[i] == 'sym':
                sym_idx.append(i)
            elif self.clnt_mode[i] == 'asym':
                asym_idx.append(i)
            else:
                clean_idx.append(i)
        return clean_idx,asym_idx,sym_idx

    def model_dist(self,w_1, w_2):
        self.set_model(w_1)
        w_1 = copy.deepcopy(self._model.state_dict())
        self.set_model(w_2)
        w_2 = copy.deepcopy(self._model.state_dict())
        assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dict with different keys"
        dist_total = torch.zeros(1).float()
        for key in w_1.keys():
            if "int" in str(w_1[key].dtype):
                continue
            dist = torch.norm(w_1[key] - w_2[key])
            dist_total += dist.cpu()

        return dist_total.cpu().item()

    def FedAvg(self, packs):
        parameters_list = [elem[0] for elem in packs]
        data_lens = [elem[1] for elem in packs]
        weights = data_lens
        sum_weight = np.sum(weights)
        weights = weights / sum_weight
        w_avg = parameters_list[0]*weights[0]
        for i in range(1,len(packs)):
            w_avg += parameters_list[i]*weights[i]
        self.round += 1
        self.set_model(w_avg)

    # def FedAvg(self,packs):
    #     round = self.round
    #     parameters_list = [elem[0] for elem in packs]
    #     for pack in packs:
    #         self.load(pack)
    #     clean_clients, asym_clients, sym_clients = self.real_noise_clnt()
    #     noisy_clients = asym_clients + sym_clients
    #     distance = np.zeros(len(packs))
    #     for n_idx in noisy_clients:
    #         dis = []
    #         for c_idx in clean_clients:
    #             dis.append(self.model_dist(parameters_list[n_idx], parameters_list[c_idx]))
    #         distance[n_idx] = min(dis)
    #     distance = distance / distance.max()
    #     self.plot_distance(distance,clean_clients, asym_clients,sym_clients)



    def evaluate(self):
        round = self.round
        self._model.eval()
        test_loader = self.dataset.get_dataloader(train=False, batch_size=self.batch_size)
        multimodel = hasattr(self._model, "models")
        loss_, acc_ = misc.evaluate(
            self._model,
            nn.CrossEntropyLoss(),
            test_loader,
            self.device,
            multimodel=multimodel,
        )
        self.logger.debug("\n|Global Epoch #%d\t Accuracy: %.2f%%\n" % (self.round-1, 100 * acc_))
        return loss_, acc_


class FedAvgESServerHandler(FedAvgServerHandler):
    def __init__(
            self,
            model: torch.nn.Module,
            global_round: int,
            sample_ratio: float,
            nll_name: str = None,
            cuda: bool = True,
            device: str = None,
            loggerfile=None,
            args=None,
            estimate_quality=None,
    ):
        FedAvgServerHandler.__init__(
            self, model, global_round, sample_ratio,cuda, device,loggerfile,args
        )
        self.args = args
        # load true noise ratio
        self.estimate_quality = np.ones(args.num_clients)
        nll_filename = f"{self.nll_name}_seed_{args.seed}_setting.pt"
        nll_file_path = os.path.join(self.args.data_dir, nll_filename)
        fednll_scene = torch.load(nll_file_path)
        if args.dataset == 'gcommand':
            self.sparse_level_clnt = fednll_scene['sparse_level_clnt']
        self.cmp_out_dir = os.path.dirname(loggerfile)
        self.ablation_txt = os.path.join(self.cmp_out_dir,'confusion_mat.txt')

        self.each_noise_ratio = fednll_scene['noise_ratio']
        if self.args.noise_mode in ['uniform', 'beta']:
            self.clnt_mode = fednll_scene['clnt_mode']
            self.prclnt_mode = {}
        elif self.args.noise_mode == 'quasireal':
            self.clnt_mode = self.sparse_level_clnt
        print('Each noise ratio is :',self.each_noise_ratio)
        print('File storage path is:', self.cmp_out_dir)
        #logger file
        writer = SummaryWriter(comment='scalar')
        self.logger.debug(writer.log_dir.replace('runs/', ''))
        self.ylim = -1
        self.clnt_loss_small_warm = [[] for ep in range(self.args.begin)]
        self.clnt_loss_small_cold = [[] for ep in range(self.args.begin)]


    @property
    def model_parameters(self) -> torch.Tensor:
        return misc.serialize_model(self._model)

    def set_model(self, parameters: torch.Tensor):
        misc.deserialize_model(self._model, parameters)

    def setup_dataset(self, dataset) -> None:
        self.dataset = dataset

    def metric_nan_solver(self,metrics,norm=True):
        # nan值用该列最小值替换
        for i in range(metrics.shape[0]):
            for j in range(metrics.shape[1]):
                if np.isnan(metrics[i, j]):
                    metrics[i, j] = np.nanmin(metrics[:, j])
        nan_columns = np.all(np.isnan(metrics), axis=0)
        metrics[:, nan_columns] = 0
        # 按列归一化
        if norm:
            for j in range(metrics.shape[1]):
                diff = metrics[:, j].max() - metrics[:, j].min()
                if diff == 0:
                    metrics[:, j] = 0
                else:
                    metrics[:, j] = (metrics[:, j] - metrics[:, j].min()) / \
                                    (metrics[:, j].max() - metrics[:, j].min())
        return metrics

    def print_recall(self,noisy_clients,small_='clean',large_='noisy'):
        sym_correct = 0
        sym_num = 0
        asym_correct = 0
        asym_num = 0
        for i in range(self.args.num_clients):
            if self.clnt_mode[i] == 'sym':
                sym_num += 1
                if i in noisy_clients:
                    sym_correct += 1
            elif self.clnt_mode[i] == 'asym':
                asym_num += 1
                if i in noisy_clients:
                    asym_correct += 1
        sym_recall = sym_correct / sym_num
        asym_recall = asym_correct / asym_num
        print('sym_recall:{:.2f} asym_recall:{:.2f}'.format(sym_recall, asym_recall))
        print('id||mode||pred mode================================')
        correct_nm_noise = 0
        noise_mode = ['sym','asym']
        count_dict = {}
        vote = []
        for i in range(self.args.num_clients):
            if i in noisy_clients:
                if self.each_noise_ratio[i] > 0:
                    correct_nm_noise += 1
                else:
                    vote.append([self.clnt_mode[i],large_])
                    print(i, self.clnt_mode[i], large_)
            else:
                if self.each_noise_ratio[i] > 0:
                    vote.append([self.clnt_mode[i],small_])
                    print(i, self.clnt_mode[i], small_)
        for sublist in vote:
            sublist_tuple = tuple(sublist)
            if sublist_tuple in count_dict:
                count_dict[sublist_tuple] += 1
            else:
                count_dict[sublist_tuple] = 1
        for key,value in count_dict.items():
            print(f'key: {key}, value: {value}')
        print('correct recall:{:.2f}'.format(correct_nm_noise / (sym_num + asym_num)))



    def clnt_2gmm(self,metrics,original_idx,criterion='loss'):
        self.metric_nan_solver(metrics)
        vote = []
        gmm_probs = np.zeros((9,len(metrics)))
        for i in range(9):
            gmm = GaussianMixture(n_components=2, random_state=i).fit(metrics)
            gmm_pred = gmm.predict(metrics)
            max_mean_index = np.argmax(gmm.means_.sum(1))# 均值最高的component坐标
            large_clients = np.where(gmm_pred == max_mean_index)[0]
            large_clients = set(list(large_clients))
            vote.append(large_clients)
            gmm_prob = gmm.predict_proba(metrics)[:, max_mean_index]  # predict_proba原理就是选择概率最高的类别
            gmm_probs[i] = gmm_prob
        gmm_probs_mean = gmm_probs.mean(axis=0)
        cnt = []
        for i in vote:
            cnt.append(vote.count(i))
        large_clients = list(vote[cnt.index(max(cnt))])
        user_id = list(range(len(original_idx)))
        small_clients = list(set(user_id) - set(large_clients))
        small_idx = [original_idx[i] for i in small_clients]
        large_idx = [original_idx[i] for i in large_clients]
        self.logger.debug('small_{}:{}'.format(criterion,small_idx))
        self.logger.debug('large_{}:{}'.format(criterion,large_idx))
        small_dict = {cid:self.clnt_mode[cid] for cid in small_idx}
        large_dict = {cid:self.clnt_mode[cid] for cid in large_idx}
        # self.print_recall(large_idx)
        y_true = []
        clnt_mode_dic = {'clean':0,'asym':1,'sym':2}
        for i in user_id:
            y_true.append(clnt_mode_dic[self.clnt_mode[i]])

        # if criterion == 'loss':
        #     y_pred = [0 if i in small_idx else 1 for i in user_id]
        # else:
        #     y_pred = [1 if i in small_idx else 0 for i in user_id]

        # conf_matrix = confusion_matrix(y_true, y_pred)
        # # Optionally, use pandas for a prettier display
        # conf_matrix_df = pd.DataFrame(conf_matrix, index=[0,1], columns=[0,1])
        # with open(self.ablation_txt, 'a') as f:
        #     print("\nGMM-2 Confusion Matrix for {}:".format(criterion),file=f)
        #     print(conf_matrix_df,file=f)
        return small_dict,large_dict,gmm_probs_mean

    def classify_clients(self,score,user_id):
        self.metric_nan_solver(score)
        gmm = GaussianMixture(n_components=3).fit(score)
        gmm_pred = gmm.predict(score)
        sorted_indx = np.argsort(gmm.means_.sum(1))
        sorted_indx = sorted_indx[::-1]
        sym_clients = np.where(gmm_pred == sorted_indx[2])[0]
        asym_clients = np.where(gmm_pred == sorted_indx[1])[0]
        clean_clients = np.where(gmm_pred == sorted_indx[0])[0]
        y_true,y_pred = [],[]
        clnt_mode_dic = {'clean': 0, 'asym': 1, 'sym': 2}
        for i in user_id:
            y_true.append(clnt_mode_dic[self.clnt_mode[i]])
            if i in sym_clients:
                y_pred.append(2)
            elif i in clean_clients:
                y_pred.append(0)
            else:
                y_pred.append(1)
        conf_matrix = confusion_matrix(y_true, y_pred)
        display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        display.plot(cmap=plt.cm.Blues)
        plt.show()
        gmm_prob = gmm.predict_proba(score)  # predict_proba原理就是选择概率最高的类别
        debug = True

    def plot_XY_patterns(self,conf_np,
                            loss_np,
                            eps: float = 1e-10,
                            figsize: tuple = (10, 8)):

        # # Calculate indicators
        # indicators = torch.log(confidences + eps) / torch.log(losses + eps)
        #
        # # Convert tensors to numpy for plotting
        # conf_np = confidences.cpu().numpy()
        # loss_np = losses.cpu().numpy()
        # ind_np = indicators.cpu().numpy()
        #
        # # Define thresholds for classification
        # # These thresholds should be tuned based on your specific data
        # clean_threshold = np.percentile(ind_np, 70)  # top 30% are clean
        # sym_threshold = np.percentile(ind_np, 30)  # bottom 30% are symmetric
        #
        # # Create client type masks
        # clean_mask = ind_np >= clean_threshold
        # sym_mask = ind_np <= sym_threshold
        # asym_mask = ~(clean_mask | sym_mask)
        clean_mask = [True if self.clnt_mode[i]=='clean' else False for i in range(self.args.num_clients)]
        sym_mask = [True if self.clnt_mode[i]=='sym' else False for i in range(self.args.num_clients)]
        asym_mask = [True if self.clnt_mode[i]=='asym' else False for i in range(self.args.num_clients)]

        # Create scatter plot
        plt.figure(figsize=figsize)

        # Plot each type with different colors and labels
        plt.scatter(conf_np[clean_mask], loss_np[clean_mask],
                    c='green', label='Clean', alpha=0.7)
        plt.scatter(conf_np[sym_mask], loss_np[sym_mask],
                    c='red', label='Symmetric Noise', alpha=0.7)
        plt.scatter(conf_np[asym_mask], loss_np[asym_mask],
                    c='blue', label='Asymmetric Noise', alpha=0.7)

        plt.xlabel('Confidence')
        plt.ylabel('Loss')
        plt.title('Client Noise Pattern Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.cmp_out_dir, 'asym_rate={}.png'.format(self.args.asym_rate)), bbox_inches='tight')

        # Add indicator value contours
        x_range = np.linspace(-0.2, 1.2, 100)
        y_range = np.linspace(np.min(loss_np)-0.2, np.max(loss_np)+0.2, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.log(X + eps) / np.log(Y + eps)
        xydic = {'Z':self.args.model,'X':conf_np, 'Y':loss_np,'pattern':self.clnt_mode}
        with open(os.path.join('plot_FedTD','XY_{}_niid.pkl'.format(self.args.model)),'wb') as f:
            pickle.dump(xydic,f)
        # Plot contour lines
        plt.contour(X, Y, Z, levels=10, colors='gray', alpha=0.3)

        plt.tight_layout()
        plt.show()
        debug = True

    def dual_gmm_PANEL(self,small_conf,small_loss,large_conf,large_loss):
        pattern_true = []
        pattern_pred = []
        mapping = {'clean':0,'asym':1,'sym':2,'none':3}
        for ix in range(self.args.num_clients):
            if ix in large_conf.keys() and ix in small_loss.keys():
                self.prclnt_mode[ix] = 'clean'
            elif ix in small_conf.keys() and ix in large_loss.keys():
                self.prclnt_mode[ix] = 'sym'
            elif ix in small_conf.keys() and ix in small_loss.keys():
                self.prclnt_mode[ix] = 'asym'
            else:#large model large conf
                self.prclnt_mode[ix] = 'none'
            # else:
            #     self.prclnt_mode[ix] = 'asym'
            pattern_true.append(mapping[self.clnt_mode[ix]])
            pattern_pred.append(mapping[self.prclnt_mode[ix]])
        mat = confusion_matrix(pattern_true,pattern_pred)
        display = ConfusionMatrixDisplay(confusion_matrix=mat)
        display.plot(cmap=plt.cm.Blues)
        plt.savefig(os.path.join(self.cmp_out_dir, 'Dual_Gmm_ep={}.png'.format(self.round)), bbox_inches='tight')
        plt.show()

        return self.prclnt_mode

    def clnt_3gmm(self,metrics,original_idx,criterion='loss'):
        self.metric_nan_solver(metrics)
        vote = []
        rst = []
        for i in range(9):
            gmm = GaussianMixture(n_components=3, random_state=i).fit(metrics)
            gmm_pred = gmm.predict(metrics)
            gmm_mean_sum = gmm.means_.sum(1)
            if criterion == 'loss':
                clean_idx = np.argsort(gmm_mean_sum)[0]
                asy_idx = np.argsort(gmm_mean_sum)[1]
                sym_idx = np.argsort(gmm_mean_sum)[2]
            else:
                clean_idx = np.argsort(gmm_mean_sum)[2]
                asy_idx = np.argsort(gmm_mean_sum)[1]
                sym_idx = np.argsort(gmm_mean_sum)[0]
            clean_clients = np.where(gmm_pred == clean_idx)[0]
            asy_clients = np.where(gmm_pred == asy_idx)[0]
            sym_clients = np.where(gmm_pred == sym_idx)[0]
            noisy_clients = list(sym_clients)+list(asy_clients)
            noisy_clients = set(list(noisy_clients))
            vote.append(noisy_clients)
            rst.append([clean_clients,asy_clients,sym_clients])
        cnt = []
        for i in vote:
            cnt.append(vote.count(i))
        noisy_clients = list(vote[cnt.index(max(cnt))])
        asy_clients = list(rst[cnt.index(max(cnt))][1])
        sym_clients = list(rst[cnt.index(max(cnt))][2])
        user_id = list(range(len(original_idx)))
        clean_clients = list(set(user_id) - set(noisy_clients))
        clean_idx = original_idx[clean_clients]
        noisy_idx = original_idx[noisy_clients]
        asy_idx = original_idx[asy_clients]
        sym_idx = original_idx[sym_clients]
        clean_dict = {cid:self.clnt_mode[cid] for cid in clean_idx}
        noisy_dict = {cid:self.clnt_mode[cid] for cid in noisy_idx}
        asy_dict = {cid:self.clnt_mode[cid] for cid in asy_idx}
        sym_dict = {cid:self.clnt_mode[cid] for cid in sym_idx}
        self.print_recall(noisy_idx)
        # Dictionary to map mode to numeric labels
        mode_mapping = {'clean': 0, 'asym': 1, 'sym': 2}

        # Generate y_true and y_pred using list comprehensions
        y_true = [mode_mapping[self.clnt_mode[i]] for i in user_id]
        y_pred = [0 if i in clean_dict else 1 if i in asy_dict else 2 for i in user_id]

        conf_matrix = confusion_matrix(y_true, y_pred)
        # Optionally, use pandas for a prettier display
        labels = np.unique(self.clnt_mode)
        conf_matrix_df = pd.DataFrame(conf_matrix, index=[0,1,2], columns=[0,1,2])
        with open(self.ablation_txt, 'a') as f:
            print("\nGMM-3 Confusion Matrix for {}:".format(criterion), file=f)
            print(conf_matrix_df, file=f)
        return clean_dict,noisy_dict

    def class_count(self,class1_data, class2_data):
        class1_counts = {'sym': 0, 'asym': 0, 'clean': 0}
        class2_counts = {'sym': 0, 'asym': 0, 'clean': 0}
        for mode in class1_data.values():
            class1_counts[mode] += 1

        for mode in class2_data.values():
            class2_counts[mode] += 1
        return class1_counts,class2_counts

    def plot_NDIcorr(self,probs,probs2):
        NDI = probs2/probs # loss/conf represents the Noise Difficulty Index

        # 创建一个DataFrame
        df = pd.DataFrame({'NDI': NDI, 'Sparse Level': self.sparse_level_clnt})
        # 计算相关性矩阵
        corr_matrix = df.corr()
        # 绘制热力图
        plt.figure(figsize=self.ploter.figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=['NDI'], yticklabels=['Sparse Level'],annot_kws={"size": self.ploter.labelsize})
        plt.savefig(os.path.join(self.cmp_out_dir,'NDIcorr_hp_{}.png'.format(self.round)))
        plt.show()

        # 绘制散点图
        sns.scatterplot(x='NDI', y='Sparse Level', data=df)
        plt.xlabel('NDI',fontsize=self.ploter.labelsize)
        plt.ylabel('Sparse Level',fontsize=self.ploter.labelsize)
        plt.savefig(os.path.join(self.cmp_out_dir,'NDIcorr_st_{}.png'.format(self.round)))
        plt.show()

    def clnt_loss_tab(self,metrics1,metrics2,sampled_clients):
        with open(self.ablation_txt,'a') as f:
            print('===============Round {}================='.format(self.round),file=f)
        #===========confidence=============================
        small, large, probs = self.clnt_2gmm(metrics1, sampled_clients,criterion='conf')
        #===========loss=============================
        small2, large2, probs2 = self.clnt_2gmm(metrics2, sampled_clients,criterion='loss')
        # self.plot_NDIcorr(probs,probs2)
        self.plot_XY_patterns(metrics1,metrics2)
        pred_clnt_mode = self.dual_gmm_PANEL(small,small2,large,large2)
        return pred_clnt_mode


    def plot_bars(self,class1_counts, class2_counts,filename):
        # Count occurrences of each mode for each class

        self.ploter = plotresult()

        # Define modes and their colors
        modes = ['clean', 'asym', 'sym']
        colors = ['r', 'g', 'b']

        # Plot bars
        fig, ax = plt.subplots(figsize=self.ploter.figsize)
        bar_width = 0.35
        index = range(len(modes))

        rects1 = ax.bar(index, [class1_counts[mode] for mode in modes], bar_width, color='green', label='Class 1')
        rects2 = ax.bar([i + bar_width for i in index], [class2_counts[mode] for mode in modes], bar_width, color='red',
                        label='Class 2')


        ax.set_xlabel('Mode', fontsize=self.ploter.labelsize)
        ax.set_ylabel('Count',fontsize=self.ploter.labelsize)
        ax.set_title(filename, fontsize=self.ploter.titlesize)
        ax.set_xticks([i + bar_width / 2 for i in index])
        plt.yticks(fontsize=self.ploter.ticksize)
        ax.set_xticklabels(modes,fontsize=self.ploter.labelsize)
        ax.legend(fontsize=self.ploter.legsize)
        if self.ylim != -1:
            plt.ylim(0,self.ylim)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cmp_out_dir,filename))
        plt.show()


    def real_noise_clnt(self):
        clean_idx = []
        noisy_idx = []
        asym_idx = []
        sym_idx = []
        for i in range(self.args.num_clients):
            if self.clnt_mode[i] == 'sym':
                sym_idx.append(i)
            elif self.clnt_mode[i] == 'asym':
                asym_idx.append(i)
            else:
                clean_idx.append(i)
        return clean_idx,asym_idx,sym_idx


    def clnt_selection_double_metr(self,metrics_ce,metrics_sce):
        ce_small,ce_large = self.clnt_2gmm(metrics_ce,np.arange(self.args.num_clients))
        sce_small,sce_large = self.clnt_2gmm(metrics_sce,np.arange(self.args.num_clients))
        clean_idx,asym_idx,sym_idx = {},{},{}
        correct_num = 0
        for i in range(self.args.num_clients):
            if i in ce_small and i in sce_small:
                clean_idx[i] = self.clnt_mode[i]
                if self.clnt_mode[i] == 'clean':
                    correct_num += 1
            elif i in ce_large and i in sce_large:
                sym_idx[i] = self.clnt_mode[i]
                if self.clnt_mode[i] == 'sym':
                    correct_num += 1
            else:
                asym_idx[i] = self.clnt_mode[i]
                if self.clnt_mode[i] == 'asym':
                    correct_num += 1
        correct_ratio = correct_num/self.args.num_clients
        print('correct_ratio',correct_ratio)

        ce_small_counts,ce_large_counts = self.class_count(ce_small,ce_large)
        sce_small_count,sce_large_counts = self.class_count(sce_small,sce_large)
        ylim1 = max(max(ce_small_counts.values()),max(ce_large_counts.values()))
        ylim2 = max(max(sce_small_count.values()),max(sce_large_counts.values()))
        self.ylim = max(ylim1, ylim2)

        self.plot_bars(ce_small_counts,ce_large_counts,'ce_gmm.png')
        self.plot_bars(sce_small_count,sce_large_counts,'sce_gmm.png')

        return list(clean_idx.keys()),list(asym_idx.keys()),list(sym_idx.keys())



    def model_dist(self,w_1, w_2):
        self.set_model(w_1)
        w_1 = copy.deepcopy(self._model.state_dict())
        self.set_model(w_2)
        w_2 = copy.deepcopy(self._model.state_dict())
        assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dict with different keys"
        dist_total = torch.zeros(1).float()


        for key in w_1.keys():
            if "int" in str(w_1[key].dtype):
                continue
            # dist = nn.functional.cosine_similarity(w_1[key], w_2[key],dim=0)
            # 提取并展平参数
            params_n = [param.view(-1) for param in w_1[key]]
            params_c = [param.view(-1) for param in w_2[key]]
            # 连接所有参数，形成一个大的一维张量
            vector_n = torch.cat(params_n)
            vector_c = torch.cat(params_c)
            # 计算余弦相似度
            dist = nn.functional.cosine_similarity(vector_n.unsqueeze(0), vector_c.unsqueeze(0), dim=1)
            print(w_1[key].shape, w_2[key].shape)
            dist_total += dist.cpu()

        return dist_total.cpu().item()



    def global_update(self, buffer):
        parameters_list = [elem[0] for elem in buffer]
        weights = [elem[1] for elem in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        self.set_model(serialized_parameters)
        # self._LOGGER.info(
        #     f"Round [{self.round}/{self.global_round}] server global update done."
        # )




    def plot_distance(self,distance,clean_clients, asym_clients,sym_clients):
        plt.figure(figsize=(10, 6))
        noisy_clients = asym_clients+sym_clients
        all_clients = list(range(self.args.num_clients))
        plt.scatter(all_clients, distance, c='gray', label='Clean Clients')
        plt.scatter(noisy_clients, distance[noisy_clients], c='red', label='Noisy Clients')
        plt.scatter(asym_clients, distance[asym_clients], c='blue',
                    label='Asymmetric Noisy Clients')
        plt.scatter(sym_clients, distance[sym_clients], c='green',
                    label='Symmetric Noisy Clients')

        plt.xlabel('Client Index')
        plt.ylabel('Normalized Distance')
        plt.legend()
        plt.title('Visualization of Client Distances')
        plt.tight_layout()
        plt.savefig(os.path.join(self.cmp_out_dir, 'distance.png'), bbox_inches='tight')

    def RoAgg(self, packs):
        parameters_list = [elem[0] for elem in packs]
        client_weight = [elem[1] for elem in packs]
        client_weight = client_weight / np.sum(client_weight)
        distance = np.zeros(len(packs))
        clean_clients = [key for key,val in self.prclnt_mode.items() if val == 'clean']
        noisy_clients = list(set(range(self.args.num_clients))-set(clean_clients))
        for n_idx in noisy_clients:
            dis = []
            for c_idx in clean_clients:
                dis.append(self.model_dist(parameters_list[n_idx], parameters_list[c_idx]))
            distance[n_idx] = min(dis)
        distance = distance / distance.max()
        client_weight = client_weight * np.exp(-distance)
        client_weight = client_weight / np.sum(client_weight)

        # print(client_weight)
        w_avg = parameters_list[0] * client_weight[0]
        for i in range(1, len(packs)):
            w_avg += parameters_list[i] * client_weight[i]
        self.set_model(w_avg)

    def NoAvg(self, packs):
        self.old_model_state = copy.deepcopy(self._model.state_dict())
        parameters_list = [elem[0] for elem in packs]
        data_lens = [elem[1] for elem in packs]
        quality = np.array([1-e for e in self.each_noise_ratio])
        weights = data_lens * quality
        sum_weight = np.sum(weights)
        weights = weights / sum_weight
        w_avg = parameters_list[0]*weights[0]
        for i in range(1,len(packs)):
            w_avg += parameters_list[i]*weights[i]
        self.set_model(w_avg)


    def FedAvg(self, packs):
        parameters_list = [elem[0] for elem in packs]
        data_lens = [elem[1] for elem in packs]
        weights = data_lens
        sum_weight = np.sum(weights)
        weights = weights / sum_weight
        w_avg = parameters_list[0]*weights[0]

        for i in range(1,len(packs)):
            w_avg += parameters_list[i]*weights[i]

        self.set_model(w_avg)



    def evaluate(self,rndplus=True):
        self._model.eval()
        test_loader = self.dataset.get_dataloader(train=False, batch_size=self.batch_size)
        multimodel = hasattr(self._model, "models")
        loss_, acc_ = misc.evaluate(
            self._model,
            nn.CrossEntropyLoss(),
            test_loader,
            self.device,
            multimodel=multimodel,
        )
        self.logger.debug("\n|Global Epoch #%d\t Accuracy: %.2f%%\n" % (self.round, 100 * acc_))
        self.round = self.round+1 if rndplus else self.round
        return loss_, acc_
