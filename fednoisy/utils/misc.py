import numpy as np
import random
import os
import json
import ast
import torch
import matplotlib.pyplot as plt
import pickle
import matplotlib.lines as mlines
import sys
from collections import Counter
from torch.utils.data._utils.collate import default_collate

class AverageMeter(object):
    """Compute and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class WatchSystematicNoise(object):
    def __init__(self,num_clients) -> None:
        self.event = [[] for i in range(num_clients)]
        self.margin = [[] for i in range(num_clients)]
        self.shift = [[] for i in range(num_clients)]
        self.event_rounds = [{'clean':[],'sym':[],'asym':[]} for i in range(num_clients)]
        self.predict_last = [[] for i in range(num_clients)]
        self.memorize_ratio_list = [[] for i in range(num_clients)]
        self.legsize = 60
        self.labelsize = 70
        self.ticksize = 70
        self.titlesize = 70
        self.figsize = (22, 16)
        self.boarderlinew = 5
        self.color = ['#17becf', '#ff7f0e', '#2ca02c',
                      '#4275AF', '#9467bd', '#8c564b',
                      '#e377c2', '#F57F13', '#d62728',
                      '#ffbb78',
                      '#aec7e8', '#17becf', '#1f77b4',
                      '#1f77b4', '#17becf', '#ff7f0e',
                      '#17becf', '#9467bd', 'aec7e8',
                      '#d62728', '#9467bd', '#d62728', 'blue']
        self.line_style = ['--','-.',':','-']


    def init_client(self,g_round,cid,datasize):
        self.cid = cid
        if g_round==0:
            self.event[cid] = np.zeros(datasize)
            self.predict_last[cid] = np.zeros(datasize)



    def update_event(self,pred_result):
        for i,(j,k) in enumerate(zip(pred_result,self.predict_last[self.cid])):
            if k>j:
                self.event[self.cid][i] += 1
        self.predict_last[self.cid] = pred_result


    def memorize_ratio(self,pred_labels,labels,mask):
        memorized_nm_noise = 0
        memorized_nm_clean = 0
        for i,(p_history,l) in enumerate(zip(pred_labels,labels)):
            # Counting the occurrences of each number
            unique_number,counts = torch.unique(p_history,return_counts=True)
            # Get the index of the maximum count
            max_count_index = counts.argmax()
            # The most frequent number
            most_frequent_number = unique_number[max_count_index]
            # Finding the most frequent number
            if most_frequent_number == l:
                if mask[i]:
                    memorized_nm_noise += 1
                else:
                    memorized_nm_clean += 1
        noise_num = torch.sum(mask)
        memratio_noise = memorized_nm_noise/torch.sum(mask) if noise_num!=0 else torch.tensor(0)
        memratio_clean = memorized_nm_clean/(len(mask)-torch.sum(mask)) if noise_num!=len(mask) else torch.tensor(0)
        self.memorize_ratio_list[self.cid].append((memratio_clean,memratio_noise))

    def update_shift(self,l_round,logits,labels):
        margin = np.zeros(len(labels))
        logits = torch.softmax(logits,dim=1)
        for i, (p,k) in enumerate(zip(logits,labels)):
            margin[i] = max(p)-p[k]
        if l_round == 0:
            self.margin[self.cid] = margin
        else:
            self.shift[self.cid] = margin-self.margin[self.cid]

    def to_bool(self,arr):
        return [True if a==1 else False for a in arr]

    def load_eventNm(self,savepath):
        data_path = os.path.join(savepath, 'event_rounds_{}.pkl'.format(self.cid))
        with open(data_path, 'rb') as f:
            self.event_rounds = pickle.load(f)
        #========for history error of not keeping average value=============
        data_path = os.path.join(savepath, 'event_info_{}.pkl'.format(self.cid))
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        self.event_rounds[self.cid]['clean'] = [e/data_dict['clean'].sum() for e in self.event_rounds[self.cid]['clean']]
        self.event_rounds[self.cid]['sym'] = [e/data_dict['sym'].sum() for e in self.event_rounds[self.cid]['sym']]
        self.event_rounds[self.cid]['asym'] = [e/data_dict['asym'].sum() for e in self.event_rounds[self.cid]['asym']]
        self.plot_eventNm(len(self.event_rounds[self.cid]['clean'])-1,savepath)



    def curr_eventNm(self,g_round,clean,sym,asym,savepath):
        event = self.event[self.cid]
        data_group1 = event[self.to_bool(clean)]
        data_group2 = event[self.to_bool(sym)]
        data_group3 = event[self.to_bool(asym)]
        clean_event_num = data_group1.sum()
        sym_event_num = data_group2.sum()
        asym_event_num = data_group3.sum()
        self.event_rounds[self.cid]['clean'].append(clean_event_num/clean.sum())
        self.event_rounds[self.cid]['sym'].append(sym_event_num/sym.sum())
        self.event_rounds[self.cid]['asym'].append(asym_event_num/asym.sum())

        data_path = os.path.join(savepath, 'event_rounds_{}.pkl'.format(self.cid))
        with open(data_path, 'wb') as f:
            pickle.dump(self.event_rounds, f)
        self.plot_eventNm(g_round,savepath)

    def plot_tstLoss(self,SYM_LOSS,ASYM_LOSS,CLEAN_LOSS,savepath):

        ax = plt.figure(figsize=self.figsize).add_subplot(111)
        x = range(len(CLEAN_LOSS))
        p, = plt.plot(x, CLEAN_LOSS, lw=8, color=self.color[0],
                      linestyle=self.line_style[0], label='Clean data',
                      fillstyle='none', markeredgewidth=5,
                      # marker=marker[method[m]], markersize=20
                      )
        p, = plt.plot(x, SYM_LOSS, lw=8,
                      color=self.color[1],
                      linestyle=self.line_style[1], label='Random noise',
                      fillstyle='none', markeredgewidth=5,
                      # marker=marker[method[m]], markersize=20
                      )
        p, = plt.plot(x, ASYM_LOSS, lw=8,
                      color=self.color[2],
                      linestyle=self.line_style[2], label='Systematic noise',
                      fillstyle='none', markeredgewidth=5,
                      # marker=marker[method[m]], markersize=20
                      )

        plt.xlabel('Rounds', fontsize=self.labelsize)
        plt.ylabel('Loss', fontsize=self.labelsize)
        plt.xticks(np.arange(0,len(CLEAN_LOSS),20,dtype=int), fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.legend(fontsize=self.legsize, loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, 'Tst_loss.png'), bbox_inches='tight')
        print('save png : {}'.format('Tst_loss.png'))
        plt.show()

    def plot_memRatio(self,mem_clean,mem_noise,savepath_all,start):

        ax = plt.figure(figsize=self.figsize).add_subplot(111)
        x = np.arange(start,start+len(mem_clean))
        plt.plot(x, mem_clean, color='blue', lw=8)
        # Filling color under the line
        plt.fill_between(x, mem_clean, color='skyblue', alpha=0.3)

        # Plotting the line
        plt.plot(x, mem_noise, color='red', lw=8)
        # Filling color under the line
        plt.fill_between(x, mem_noise, color='#FF00AA', alpha=0.3)


        plt.xlabel('Rounds', fontsize=self.labelsize)
        plt.ylabel('Memorization Ratio', fontsize=self.labelsize)
        plt.xticks(np.arange(0,len(mem_clean),20,dtype=int), fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.legend(fontsize=self.legsize, loc='best')
        plt.tight_layout()
        plt.savefig(savepath_all,bbox_inches='tight')
        print('save png : {}'.format(os.path.basename(savepath_all)))
        plt.show()

    def plot_trclntLoss(self,TR_Loss,clnt_mode,savepath):

        ax = plt.figure(figsize=self.figsize).add_subplot(111)
        TR_Loss = np.array(TR_Loss)
        TR_Loss = np.transpose(TR_Loss)
        round_num = TR_Loss.shape[1]
        round_num = 50
        x = range(round_num)
        pattern_dic = {'clean':'clean clients','asym':'structured noisy clients','sym':'random noisy clients'}
        marker_dic = {'clean':'o','asym':'*','sym':'^'}
        colors = ['#FDF5E6', '#E6E6FA','#FFE4E1','#FFB6C1',
            '#FBCEB1', '#F2D2BD', '#FAD5A5', '#FFD580',  # 橙色系
            '#ADD8E6', '#B0E0E6', '#E0FFFF', '#87CEFA',  # 蓝色系

        ]
        color_dic = {'clean':colors[-2:],'asym':colors[:4],'sym':colors[4:8]}
        for idx in range(len(TR_Loss)):
            p, = plt.plot(x, TR_Loss[idx][:round_num], lw=8, color=color_dic[clnt_mode[idx]][0],
                      linestyle='-', label=pattern_dic[clnt_mode[idx]], markeredgewidth=5,
                      marker=marker_dic[clnt_mode[idx]], markersize=30,markevery=4
                      )
            color_dic[clnt_mode[idx]] = color_dic[clnt_mode[idx]][1:]
        plt.xlabel('Rounds', fontsize=self.labelsize)
        plt.ylabel('Training Loss', fontsize=self.labelsize)
        plt.xticks(np.arange(0,round_num,20,dtype=int), fontsize=self.ticksize)
        # 创建自定义图例
        handles = []
        labels = []
        plt.yticks(fontsize=self.ticksize)
        for label, marker in marker_dic.items():
            handle = mlines.Line2D([], [], color='black', marker=marker, linestyle='-', markersize=30, label=label)
            handles.append(handle)
            labels.append(pattern_dic[label])
        # legend_handles = [mlines.Line2D([], [], marker=value, linestyle='None', markersize=20, markerfacecolor='black',
        #                                 label=pattern_dic[key]) for key,value in marker_dic.items()]
        plt.legend(handles=handles, labels=labels,fontsize=self.legsize, loc='best')
        plt.tight_layout()
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=1)
        plt.savefig(savepath,bbox_inches='tight')
        print('save png : {}'.format('Tr_clntloss.png'))
        plt.show()

    def plot_trLoss(self,SYM_LOSS,ASYM_LOSS,CLEAN_LOSS,savepath):

        ax = plt.figure(figsize=self.figsize).add_subplot(111)
        x = range(len(CLEAN_LOSS))
        p, = plt.plot(x, CLEAN_LOSS, lw=8, color=self.color[0],
                      linestyle=self.line_style[0], label='Clean data',
                      fillstyle='none', markeredgewidth=5,
                      # marker=marker[method[m]], markersize=20
                      )
        p, = plt.plot(x, SYM_LOSS, lw=8,
                      color=self.color[1],
                      linestyle=self.line_style[1], label='Random noise',
                      fillstyle='none', markeredgewidth=5,
                      # marker=marker[method[m]], markersize=20
                      )
        p, = plt.plot(x, ASYM_LOSS, lw=8,
                      color=self.color[2],
                      linestyle=self.line_style[2], label='Systematic noise',
                      fillstyle='none', markeredgewidth=5,
                      # marker=marker[method[m]], markersize=20
                      )

        plt.xlabel('Rounds', fontsize=self.labelsize)
        plt.ylabel('Loss', fontsize=self.labelsize)
        plt.xticks(np.arange(0,len(CLEAN_LOSS),40,dtype=int), fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.legend(fontsize=self.legsize, loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, 'Tr_loss.png'),bbox_inches='tight')
        print('save png : {}'.format('Tr_loss.png'))
        plt.show()

    def plot_conf_metric(self,RECALL,PRECISION,savepath,aspect='sudo'):
        ax = plt.figure(figsize=self.figsize).add_subplot(111)
        x = range(len(RECALL['sym']))
        for idx,mode in enumerate(RECALL.keys()):
            p, = plt.plot(x, RECALL[mode], lw=8, color=self.color[idx],
                          linestyle=self.line_style[0], label='recall_{}'.format(mode),
                          fillstyle='none', markeredgewidth=5,
                          # marker=marker[method[m]], markersize=20
                          )
            p, = plt.plot(x, PRECISION[mode], lw=8,
                          color=self.color[idx],
                          linestyle=self.line_style[2], label='precision_{}'.format(mode),
                          fillstyle='none', markeredgewidth=5,
                          # marker=marker[method[m]], markersize=20
                          )
        plt.xlabel('Rounds', fontsize=self.labelsize)
        plt.ylabel('Metrics', fontsize=self.labelsize)
        plt.xticks(np.arange(0, len(RECALL['sym']), 20, dtype=int), fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.legend(fontsize=self.legsize, loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, '{}_Confmetric.png'.format(aspect)),bbox_inches='tight')
        plt.show()
        print('save png : {}'.format('{}_Confmetric.png'.format(aspect)))





    def plot_trLoss_examp(self,SYM_LOSS,ASYM_LOSS,CLEAN_LOSS,savepath):

        ax = plt.figure(figsize=self.figsize).add_subplot(111)
        x = range(len(CLEAN_LOSS))
        p, = plt.plot(x, CLEAN_LOSS, lw=8, color=self.color[0],
                      linestyle=self.line_style[0], label='Clean data',
                      fillstyle='none', markeredgewidth=5,
                      # marker=marker[method[m]], markersize=20
                      )
        p, = plt.plot(x, SYM_LOSS, lw=8,
                      color=self.color[1],
                      linestyle=self.line_style[1], label='symmetric noise',
                      fillstyle='none', markeredgewidth=5,
                      # marker=marker[method[m]], markersize=20
                      )
        p, = plt.plot(x, ASYM_LOSS, lw=8,
                      color=self.color[2],
                      linestyle=self.line_style[2], label='asymmetric noise',
                      fillstyle='none', markeredgewidth=5,
                      # marker=marker[method[m]], markersize=20
                      )

        plt.xlabel('Rounds', fontsize=self.labelsize)
        plt.ylabel('Global avg loss', fontsize=self.labelsize)
        plt.xticks(np.arange(0,len(CLEAN_LOSS),40,dtype=int), fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.legend(fontsize=self.legsize, loc='best')
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        plt.tight_layout()
        plt.savefig(os.path.join(savepath, 'Tr_loss_examp.png'), bbox_inches='tight')
        print('save png : {}'.format('Tr_loss_examp.png'))
        plt.show()

    def plot_eventNm(self,g_round,savepath):

        ax = plt.figure(figsize=self.figsize).add_subplot(111)
        p, = plt.plot(range(g_round + 1), self.event_rounds[self.cid]['clean'], lw=8, color=self.color[0],
                      linestyle=self.line_style[0], label='Clean data',
                      fillstyle='none', markeredgewidth=5,
                      # marker=marker[method[m]], markersize=20
                      )
        p, = plt.plot(range(g_round + 1), self.event_rounds[self.cid]['sym'], lw=8,
                      color=self.color[1],
                      linestyle=self.line_style[1], label='Random noise',
                      fillstyle='none', markeredgewidth=5,
                      # marker=marker[method[m]], markersize=20
                      )
        p, = plt.plot(range(g_round + 1), self.event_rounds[self.cid]['asym'], lw=8,
                      color=self.color[2],
                      linestyle=self.line_style[2], label='Systematic noise',
                      fillstyle='none', markeredgewidth=5,
                      # marker=marker[method[m]], markersize=20
                      )

        plt.xlabel('Rounds', fontsize=self.labelsize)
        plt.ylabel('Number of Forgetting Events', fontsize=self.labelsize)
        plt.xticks(np.arange(0,g_round + 1,20,dtype=int), fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.legend(fontsize=self.legsize, loc='best')
        plt.savefig(os.path.join(savepath, 'Curr_eventNum_{}.png'.format(self.cid)))
        print('save png : {}'.format('Curr_event.png'))
        plt.show()

    def plot_memoryHis(self,noise_mode,noise_rate,savepath):
        plt.figure(figsize=self.figsize)
        memoryrario_round = self.memorize_ratio_list[self.cid]
        mem_clean = [item[0].cpu() for item in memoryrario_round]
        mem_noise = [item[1].cpu() for item in memoryrario_round]
        x = np.arange(len(mem_clean))

        # Plotting the line
        plt.plot(x, mem_clean, color='blue', linewidth=2)
        # Filling color under the line
        plt.fill_between(x, mem_clean, color='skyblue', alpha=0.3)

        # Plotting the line
        plt.plot(x, mem_noise, color='red', linewidth=2)
        # Filling color under the line
        plt.fill_between(x, mem_noise, color='#FF00AA', alpha=0.3)

        plt.xlabel('Rounds', fontsize=self.labelsize)
        plt.ylabel('Memorization Rate', fontsize=self.labelsize)
        plt.xticks(range(min(x),max(x),5),fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.legend(fontsize=self.legsize, loc='best')
        plt.savefig(
            os.path.join(savepath, 'MemoryRate_{}_{}_r={:.2f}.png'.format(self.cid, noise_mode,noise_rate)),
            fontsize=self.labelsize)

        data_path = os.path.join(savepath, 'memory_info{}.pkl'.format(self.cid))
        with open(data_path, 'wb') as f:
            pickle.dump(self.memorize_ratio_list[self.cid], f)

        plt.show()




    def plot_eventHis(self,g_round,clean,sym,asym,savepath):
        event = self.event[self.cid]
        # Example data
        # Separate data based on groups
        data_group1 = event[self.to_bool(clean)]
        data_group2 = event[self.to_bool(sym)]
        data_group3 = event[self.to_bool(asym)]

        noise_rate = 1-clean.sum()/len(clean)
        # Saving
        data_dict = {'clean': clean, 'sym': sym, 'asym': asym, 'event': event}
        data_path = os.path.join(savepath, 'event_info_{}.pkl'.format(self.cid))
        with open(data_path, 'wb') as f:
            pickle.dump(data_dict, f)
        # Plotting
        plt.figure(figsize=self.figsize)


        plt.hist(data_group1, bins=np.arange(min(event), max(event) + 1), alpha=0.5, label='Clean data', color='blue')
        plt.hist(data_group2, bins=np.arange(min(event), max(event) + 1), alpha=0.5, label='Symmetric noise', color='orange')
        plt.hist(data_group3, bins=np.arange(min(event), max(event) + 1), alpha=0.5, label='ASymmetric noise', color='red')

        plt.xlabel('Number of Forgetting Events',fontsize=self.labelsize)
        plt.xticks(fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.legend(fontsize=self.legsize, loc='best')
        plt.savefig(os.path.join(savepath,'ForgettingEvents_{}_r={:.2f}_round={}.png'.format(self.cid,noise_rate,g_round)),fontsize=self.labelsize)

        plt.show()

    def plot_sceHis(self,sce,g_round,clean,sym,asym,savepath):
        # Example data
        # Separate data based on groups
        data_group1 = sce[self.to_bool(clean)]
        data_group2 = sce[self.to_bool(sym)]
        data_group3 = sce[self.to_bool(asym)]

        noise_rate = 1-clean.sum()/len(clean)

        # Saving
        data_dict = {'clean':clean,'sym':sym,'asym':asym,'sce':sce}
        data_path = os.path.join(savepath,'sce_info_{}.pkl'.format(self.cid))
        with open(data_path,'wb') as f:
            pickle.dump(data_dict,f)

        # Plotting
        plt.figure(figsize=self.figsize)
        plt.hist(data_group1, bins=20, alpha=0.5, label='Clean data', color='blue')
        plt.hist(data_group2, bins=20, alpha=0.5, label='Symmetric noise', color='orange')
        plt.hist(data_group3, bins=20, alpha=0.5, label='ASymmetric noise', color='red')

        plt.xlabel('SCE Loss',fontsize=self.labelsize)
        plt.xticks(fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.legend(fontsize=self.legsize, loc='best')
        plt.savefig(os.path.join(savepath,'sce-loss_{}_r={:.2f}_round={}.png'.format(self.cid,noise_rate,g_round)),fontsize=self.labelsize)

        plt.show()

    def plot_shiftHis(self,clean,sym,asym,savepath):
        shift = self.shift[self.cid]
        # Example data
        # Separate data based on groups
        data_group1 = shift[self.to_bool(clean)]
        data_group2 = shift[self.to_bool(sym)]
        data_group3 = shift[self.to_bool(asym)]

        noise_rate = 1-clean.sum()/len(clean)

        # Saving
        data_dict = {'clean':clean,'sym':sym,'asym':asym,'shift':shift}
        data_path = os.path.join(savepath,'shift_info_{}.pkl'.format(self.cid))
        with open(data_path,'wb') as f:
            pickle.dump(data_dict,f)

        # Plotting
        plt.figure(figsize=self.figsize)
        plt.hist(data_group1, bins=20, alpha=0.5, label='Clean data', color='blue')
        plt.hist(data_group2, bins=20, alpha=0.5, label='Symmetric noise', color='orange')
        plt.hist(data_group3, bins=20, alpha=0.5, label='ASymmetric noise', color='red')

        plt.xlabel('Margin Shift',fontsize=self.labelsize)
        plt.xticks(fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.legend(fontsize=self.legsize, loc='best')
        plt.savefig(os.path.join(savepath,'Margin Shift_{}_r={:.2f}.png'.format(self.cid,noise_rate)),fontsize=self.labelsize)

        plt.show()



def setup_seed(seed: int = 0):
    """
    Args:
        seed (int): random seed value.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, criterion, test_loader, device, multimodel=False):
    """Evaluate classify task model accuracy, allow ``model`` contains multiple networks .

    Returns:
        (loss.avg, acc.avg)
    """
    if multimodel is False:
        model.eval()
    else:
        for net in model.models:
            net.eval()

    loss_ = AverageMeter()
    acc_ = AverageMeter()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = len(labels)

            outputs = model(inputs)
            if multimodel is True:
                # sum over outputs of all nets
                outputs = torch.sum(torch.stack(outputs), dim=0)

            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            loss_.update(loss.item(), batch_size)
            acc_.update(torch.sum(predicted.eq(labels)).item() / batch_size, batch_size)

    return loss_.avg, acc_.avg


def save_json(file_name, root_dir, content):
    file_path = os.path.join(root_dir, file_name)
    with open(file_path, "w") as out_f:
        json.dump(content, out_f)
    return True


def make_dirs(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.mkdir(dir_path)
        except FileNotFoundError:
            os.makedirs(dir_path)


def make_alg_name(args):
    if args.criterion != "ce":
        alg_name = "FedAvg-RobustLoss"
    else:
        if args.td is True:
            alg_name = 'FedTD'
        elif args.mixup is True:
            alg_name = "FedAvg-Mixup"
        elif args.coteaching is True:
            alg_name = "FedAvg-Coteaching"
        elif args.dynboot is True:
            alg_name = "FedAvg-DynamicBootstrapping"
        elif args.favg:
            alg_name = "FedAvg"
        else:
            alg_name = 'FedES'

    return alg_name


def make_exp_name(fed_alg_name="fedavg", args=None) -> str:
    """Make logging name for federated algrithms.

    Args:
        fed_alg_name (str, optional): _description_. Defaults to "fedavg".
        args (_type_, optional): _description_. Defaults to None.

    Returns:
        str: _description_
    """
    if fed_alg_name == "fedavg":
        noisy_alg_name = None
        arch_name = f"arch={args.model}"
        opt_name = f"lr={args.lr:.4f}-momentum={args.momentum:.2f}-weight_decay={args.weight_decay:.5f}"
        criterion_name = make_criterion_name(args)
        if args.td is True:
            noisy_alg_name = f"fedtd=True-begin={args.begin}-end={args.end}-sudo={args.sudo}-LA={args.LA}"
        elif args.cdr is True:
            if args.realq:
                noisy_alg_name = f"cdr=True-realq={args.realq}-norm_mean={args.noise_ratio:.2f}-std={args.std:.2f}"
            else:
                if args.sgn:
                    if args.stage1>0:
                        noisy_alg_name = f"cdr=True-sgn={args.sgn}-dynamic={args.dynamic}-s1={args.stage1}-norm_mean={args.noise_ratio:.2f}-std={args.std:.2f}"
                    else:
                        noisy_alg_name = f"cdr=True-sgn={args.sgn}-dynamic={args.dynamic}-norm_mean={args.noise_ratio:.2f}-std={args.std:.2f}"
                elif args.grad:
                    # noisy_alg_name = f"cdr=True-grad={args.grad}-sgn={args.sgn}-norm_mean={args.noise_ratio:.2f}-std={args.std:.2f}"
                    noisy_alg_name = f"cdr=True-grad={args.grad}-sgn={args.sgn}-dynamic={args.dynamic}-s1={args.stage1}-norm_mean={args.noise_ratio:.2f}-std={args.std:.2f}"
                else:
                    noisy_alg_name = f"cdr=True-norm_mean={args.noise_ratio:.2f}-std={args.std:.2f}"
        elif args.noro is True:
            if args.oldclntsel:
                noisy_alg_name = f"noro=True-oldclntsel-LA={args.LA}-s1={args.s1}-begin={args.begin}-end={args.end}-a={args.a}"
            elif args.LA:
                noisy_alg_name = f"noro=True-s1={args.s1}-begin={args.begin}-end={args.end}-a={args.a}"
            else:
                noisy_alg_name = f"noro=True-LA={args.LA}-s1={args.s1}-begin={args.begin}-end={args.end}-a={args.a}"
        elif args.corr is True:
            noisy_alg_name = f'corr=True-frac1={args.frac1}-frac2={args.frac2}-iteration1={args.iteration1}-rounds1={args.rounds1}-rounds2={args.rounds2}-fedmixup={args.fedmixup}-beta={args.beta}'
        elif args.corroffi is True:
            noisy_alg_name = f'corroffi=True-frac1={args.frac1}-frac2={args.frac2}-iteration1={args.iteration1}-rounds1={args.rounds1}-rounds2={args.rounds2}-beta={args.beta}'
        elif args.rofl is True:
            noisy_alg_name = f"rofl=True-T_pl={args.T_pl}-lambda_cen={args.lambda_cen}-lambda_e={args.lambda_e}"
        elif args.mixup is True:
            noisy_alg_name = f"mixup=True-mixup_alpha={args.mixup_alpha:.2f}"
        elif args.coteaching is True:
            noisy_alg_name = f"coteaching=True-coteaching_forget_rate={args.coteaching_forget_rate}-coteaching_num_gradual={args.coteaching_num_gradual}-coteaching_exponent={args.coteaching_exponent}"

        # elif args.hisce is True:
        #     if args.warm:
        #         if args.perclass:
        #             noisy_alg_name = f"kdsce_sudo-warm={args.warm}-perclass"
        #         else:
        #             noisy_alg_name = f"kdsce_sudo-warm={args.warm}-Roagg-sudo_once-sudo-ce={args.sudo_ce}-clean_sce={args.clean_sce}-continu-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}"
        #     elif args.Nagg:
        #         noisy_alg_name = f"hisce=True-Nagg={args.Nagg}-LA={args.LA}-s1={args.s1}"
        #     elif args.all_sce:
        #         noisy_alg_name = f"hisce=True-allsce-Ragg-continu-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}-begin={args.begin}-end={args.end}"
        #     elif args.real_ESsce_sce:
        #         noisy_alg_name = f"hisce=True-real_ESsce_sce-ESsce-{args.ESsce}-Ragg-continu-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}-begin={args.begin}-end={args.end}"
        #     elif args.continu:
        #         if args.Ragg:
        #             if args.Roagg:
        #                 noisy_alg_name = f"kdsce_sudo-Roagg-sudo_once-sudo-ce={args.sudo_ce}-clean_sce={args.clean_sce}-Ragg-continu-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}"
        #             elif args.Roagg2:
        #                 noisy_alg_name = f"kdsce_sudo-Roagg2-sudo_once-sudo-ce={args.sudo_ce}-clean_sce={args.clean_sce}-Ragg-continu-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}"
        #             elif args.BDFA:
        #                 noisy_alg_name = f"kdsce_sudo-BDFA-sudo_once-sudo-ce={args.sudo_ce}-clean_sce={args.clean_sce}-Ragg-continu-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}"
        #             elif args.sym_sudo:
        #                 if args.sudo_once:
        #                     noisy_alg_name = f"hisce=True-kdsce_sudo-sudo_once-sudo-ce={args.sudo_ce}-clean_sce={args.clean_sce}-Ragg-continu-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}"
        #                 else:
        #                     noisy_alg_name = f"hisce=True-kdsce_sudo-sudo_ce={args.sudo_ce}-clean_sce={args.clean_sce}-Ragg-continu-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}"
        #             elif args.denoi_sce:
        #                 noisy_alg_name = f"hisce=True-denoi_sce-Ragg-continu-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}"
        #             elif args.kd_sce:
        #                 noisy_alg_name = f"hisce=True-real-kdsce_sce-Ragg-continu-alpha={args.sce_alpha}-LA={args.LA}-kdsceLA={args.kdsceLA}-sceLA={args.sceLA}"
        #             elif args.real_tsce_sce:
        #                 noisy_alg_name = f"hisce=True-real-tsce_sce-Ragg-continu-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}"
        #             elif not args.LA and not args.warm_LA and not args.sceLA:
        #                 noisy_alg_name = f"hisce=True-real-Ragg-continu-noLA-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}-begin={args.begin}-end={args.end}"
        #             else:
        #                 noisy_alg_name = f"hisce=True-real-Ragg-continu-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}-begin={args.begin}-end={args.end}"
        #         else:
        #             noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-real-continu-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}-begin={args.begin}-end={args.end}"
        #     elif args.ESsce:
        #         if args.a != 0.8:
        #             noisy_alg_name = f"hisce=True-real-Ragg-ESsce-a={args.a}-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}-begin={args.begin}-end={args.end}"
        #         elif args.end != 49:
        #             noisy_alg_name = f"hisce=True-real-Ragg-ESsce-end={args.end}-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}-begin={args.begin}-end={args.end}"
        #         else:
        #             noisy_alg_name = f"hisce=True-real-Ragg-ESsce-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}-begin={args.begin}-end={args.end}"
        #     elif args.real_sce_noro:
        #         if args.Ragg:
        #             noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-real-Ragg-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}-begin={args.begin}-end={args.end}"
        #         else:
        #             noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-real-alpha={args.sce_alpha}-LA={args.LA}-sceLA={args.sceLA}-begin={args.begin}-end={args.end}"
        #     elif args.denoi_sce:
        #         if args.Ragg:
        #             noisy_alg_name = f"hisce=True-denoi_sce-Ragg-alpha={args.sce_alpha}-LA={args.LA}-s1={args.s1}"
        #         elif args.Dagg:
        #             noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-denoi_sce-alpha={args.sce_alpha}-LA={args.LA}-s1={args.s1}"
        #     elif args.Dagg:
        #         if args.denoi_oldsce:
        #             noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-denoi_oldsce-alpha={args.sce_alpha}-LA={args.LA}-s1={args.s1}"
        #         elif args.denoi_noro:
        #             noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-denoi_noro-alpha={args.sce_alpha}-LA={args.LA}-s1={args.s1}"
        #         elif args.estsce_tscse:
        #             if args.esceOnly:
        #                 noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-estsce_tsce-esceOnly-alpha={args.sce_alpha}-LA={args.LA}-s1={args.s1}"
        #             else:
        #                 noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-estsce_tsce-alpha={args.sce_alpha}-LA={args.LA}-s1={args.s1}"
        #         elif args.es2tsce_tscse:
        #             noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-es2tsce_tsce-alpha={args.sce_alpha}-LA={args.LA}-s1={args.s1}"
        #         elif args.wu_watch:
        #             if args.confidence:
        #                 noisy_alg_name = f"hisce=True-confidence-warmup_watch-s1={args.s1}"
        #             else:
        #                 noisy_alg_name = f"hisce=True-warmup_watch-s1={args.s1}"
        #         else:
        #             if args.LA == 1:
        #                 if args.warm_LA == 0:
        #                     if args.cold_LA == 1:
        #                         if args.TsceLA:
        #                             if args.teacher_no_LA:
        #                                 noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-double_boost-alpha={args.sce_alpha}-TsceLA-esceLA-teacher_noLA-warm_LA={args.warm_LA}-cold_LA={args.cold_LA}-LA={args.LA}-s1={args.s1}"
        #                             elif args.esceLA:
        #                                 noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-double_boost-alpha={args.sce_alpha}-TsceLA-esceLA-warm_LA={args.warm_LA}-cold_LA={args.cold_LA}-LA={args.LA}-s1={args.s1}"
        #                             else:
        #                                 noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-double_boost-alpha={args.sce_alpha}-TsceLA-warm_LA={args.warm_LA}-cold_LA={args.cold_LA}-LA={args.LA}-s1={args.s1}"
        #                         elif args.sceLA:
        #                             if args.end != 49:
        #                                 noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-double_boost-alpha={args.sce_alpha}-sceLA-warm_LA={args.warm_LA}-cold_LA={args.cold_LA}-LA={args.LA}-begin={args.begin}-end={args.end}-s1={args.s1}"
        #                             else:
        #                                 noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-double_boost-alpha={args.sce_alpha}-sceLA-warm_LA={args.warm_LA}-cold_LA={args.cold_LA}-LA={args.LA}-s1={args.s1}"
        #                         else:
        #                             noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-double_boost-alpha={args.sce_alpha}-warm_LA={args.warm_LA}-cold_LA={args.cold_LA}-LA={args.LA}-s1={args.s1}"
        #                     else:
        #                         if args.TsceLA:
        #                             noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-double_boost-alpha={args.sce_alpha}-TsceLA-warm_LA={args.warm_LA}-LA={args.LA}-s1={args.s1}"
        #                         elif args.sceLA:
        #                             noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-double_boost-alpha={args.sce_alpha}-sceLA-warm_LA={args.warm_LA}-LA={args.LA}-s1={args.s1}"
        #                         else:
        #                             noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-double_boost-alpha={args.sce_alpha}-warm_LA={args.warm_LA}-LA={args.LA}-s1={args.s1}"
        #                 else:
        #                     noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-double_boost-alpha={args.sce_alpha}-LA={args.LA}-s1={args.s1}"
        #             else:
        #                 noisy_alg_name = f"hisce=True-Dagg={args.Dagg}-alpha={args.sce_alpha}-LA={args.LA}-s1={args.s1}"
        #     else:
        #         noisy_alg_name = f"hisce=True-LA={args.LA}-s1={args.s1}"
        # elif args.watch is True:
        #     noisy_alg_name = f"watch=True"
        # elif args.watchtr or args.memory or args.watchtst:
        #     noisy_alg_name = f'memory={args.memory}-watchtr={args.watchtr}-watchtst={args.watchtst}'
        # elif args.glob_memory is True:
        #     noisy_alg_name = 'glob_memory'
        # elif args.watch_conf is True:
        #     noisy_alg_name = 'watch_conf'


        other_name = f"com_round={args.com_round}-local_epochs={args.epochs}-sample_ratio={args.sample_ratio:.2f}-batch_size={args.batch_size}-seed={args.seed}"

    if noisy_alg_name is None:
        exp_name = "-".join(
            [fed_alg_name, criterion_name, arch_name, opt_name, other_name]
        )
    else:
        exp_name = "-".join(
            [
                fed_alg_name,
                criterion_name,
                noisy_alg_name,
                arch_name,
                opt_name,
                other_name,
            ]
        )

    return exp_name


def make_exp_name_centr(alg_name="dividemix", args=None):
    if alg_name == "crossentropy":
        noise_name = f"noise_mode={args.noise_mode}-noise_ratio={args.noise_ratio:.2f}"
        opt_name = f"lr={args.lr:.4f}-momentum={args.momentum:.2f}-weight_decay={args.weight_decay:.5f}"
        other_name = f"num_epochs={args.num_epochs}-batch_size={args.batch_size}-seed={args.seed}"
        exp_name = "-".join([noise_name, opt_name, other_name])

    elif alg_name == "dividemix":
        noise_name = f"noise_mode={args.noise_mode}-noise_ratio={args.noise_ratio:.2f}"
        alg_param = f"p_threshold={args.p_threshold:.2f}-lambda_u={args.lambda_u}-T={args.T:.2f}-alpha={args.alpha:.2f}"
        opt_name = f"lr={args.lr:.4f}-momentum={args.momentum:.2f}-weight_decay={args.weight_decay:.5f}"
        other_name = f"num_epochs={args.num_epochs}-batch_size={args.batch_size}-seed={args.seed}"
        exp_name = "-".join([noise_name, alg_name, alg_param, opt_name, other_name])

    elif alg_name == "coteaching":
        pass
    return exp_name


def make_criterion_name(args):
    criterion_name = f"criterion={args.criterion}"

    if args.criterion == "ce":
        criterion_param = ""
    elif args.criterion == "sce":
        criterion_param = f"sce_alpha={args.sce_alpha:.2f}-sce_beta={args.sce_beta:.2f}"
    elif args.criterion == "lsce":
        criterion_param = f"sce_alpha=data_quality-sce_beta={args.sce_beta:.2f}"
    elif args.criterion in ["rce", "nce", "nrce"]:
        criterion_param = f"loss_scale={args.loss_scale:.2f}"
    elif args.criterion == "gce":
        criterion_param = f"gce_q={args.gce_q:.2f}"
    elif args.criterion == "ngce":
        criterion_param = f"loss_scale={args.loss_scale:.2f}-gce_q={args.gce_q:.2f}"
    elif args.criterion in ["mae", "nmae"]:
        criterion_param = f"loss_scale={args.loss_scale:.2f}"
    elif args.criterion in ["focal", "nfocal"]:
        if args.focal_alpha is None:
            criterion_param = f"focal_gamma={args.focal_gamma:.2f}-focal_alpha=None"
        else:
            criterion_param = (
                f"focal_gamma={args.focal_gamma:.2f}-focal_alpha={args.focal_alpha:.2f}"
            )
    criterion_name = "-".join([criterion_name, criterion_param])
    return criterion_name


def serialize_model(model: torch.nn.Module) -> torch.Tensor:
    parameters = [param.data.view(-1) for param in model.state_dict().values()]
    m_parameters = torch.cat(parameters)
    m_parameters = m_parameters.cpu()

    return m_parameters


def deserialize_model(
    model: torch.nn.Module, serialized_parameters: torch.Tensor, mode="copy"
):
    current_index = 0  # keep track of where to read from grad_update

    for param in model.state_dict().values():
        numel = param.numel()
        size = param.size()
        if mode == "copy":
            param.copy_(
                serialized_parameters[current_index : current_index + numel].view(size)
            )
        elif mode == "add":
            param.add_(
                serialized_parameters[current_index : current_index + numel].view(size)
            )
        else:
            raise ValueError(
                'Invalid deserialize mode {}, require "copy" or "add" '.format(mode)
            )
        current_index += numel


def result_parser(result_path):
    """_summary_

    Args:
        result_path (str): _description_

    Returns:
        tuple[List[float], List[float], Dict]: _description_
    """
    with open(result_path, "r") as f:
        lines = f.readlines()
    # hist accuracy
    accs = [float(item) for item in lines[1].strip()[5:-1].split(", ")]
    # hist losses
    losses = [float(item) for item in lines[2].strip()[6:-1].split(", ")]
    # hyperparameter setting
    setting_dict = ast.literal_eval(lines[0].strip())
    return accs, losses, setting_dict
