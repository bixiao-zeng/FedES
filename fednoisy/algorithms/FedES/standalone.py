import os
import sys
import torch
import numpy as np

from fedlab.utils.logger import Logger
from fedlab.core.standalone import StandalonePipeline
from fednoisy.data.NLLData import functional as nllF
from fednoisy.utils.misc import make_dirs, make_exp_name, result_parser, make_alg_name
from fednoisy.data import CLASS_NUM

class FedAvgBase(StandalonePipeline):
    def __init__(self, handler, trainer, args, logger=None):
        super().__init__(handler, trainer)
        self.args = args
        self.exp_name = make_exp_name("fedavg", args)
        self.nll_name = nllF.FedNLL_name(**vars(args))
        alg_name = make_alg_name(args)
        self.out_path = os.path.join(args.out_dir, self.nll_name, alg_name, self.exp_name)
        make_dirs(self.out_path)
        self.record_file = os.path.join(self.out_path, "result_record.txt")
        self.best_model_path = os.path.join(self.out_path, "best_global_model.pth")
        self.last_model_path = os.path.join(self.out_path, "last_global_model.pth")
        self.loss_hist = []
        self.acc_hist = []
        self.max_acc = 0

    def _record_results(self, loss, acc):
        self.loss_hist.append(loss)
        self.acc_hist.append(acc * 100)
        with open(self.record_file, "w") as f:
            f.write(f"{vars(self.args)}\n")
            f.write("acc:" + str(self.acc_hist) + "\n")
            f.write("loss:" + str(self.loss_hist) + "\n")

    def _save_model(self, path):
        torch.save({"model": self.handler._model.state_dict(), "round": self.handler.round}, path)

    def _load_model(self, path):
        checkpoint = torch.load(path)
        self.handler._model.load_state_dict(checkpoint['model'])
        self.handler.round = checkpoint['round']

    def evaluate(self):
        loss, acc = self.handler.evaluate()
        self._record_results(loss, acc)
        if hasattr(self, "save_best") and self.save_best and acc > self.max_acc:
            self.max_acc = acc
            torch.save(self.handler._model.state_dict(), self.best_model_path)
            self._LOGGER.info("Best global model saved.")

class FedAvgStandalone(FedAvgBase):
    def __init__(self, handler, trainer, args, logger=None, save_best=False, save_last=True):
        super().__init__(handler, trainer, args, logger)
        self.save_best = save_best
        self.save_last = save_last

    def main(self):
        if self.args.watchtrloss:
            self.trainer.watch_metric()
            sys.exit()
        if os.path.exists(self.record_file):
            accs, _, _ = result_parser(self.record_file)
            if len(accs) >= self.args.com_round:
                self._LOGGER.info(f"Experiment done! Result saved in {self.record_file}!")
                return
        if self.args.load_last:
            self._load_model(self.last_model_path)
            self.evaluate()
        for rnd in range(self.args.end):
            if self.save_last:
                self._save_model(self.last_model_path)
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package
            self.trainer.local_process(broadcast, sampled_clients, self.handler.round)
            uploads = self.trainer.uplink_package
            self.handler.FedAvg(uploads)
            self.evaluate()

class FedAvgESStandalone(FedAvgBase):
    def __init__(self, handler, trainer, args, logger=None, save_best=False, save_last=False):
        super().__init__(handler, trainer, args, logger)
        self.save_best = save_best
        self.save_last = save_last

    def warm_up_stage(self, metric=True):
        model_path = os.path.join(self.out_path, 'stage1_model.pth')
        for rnd in range(self.args.begin):
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package
            self.trainer.local_process(broadcast, sampled_clients, rnd)
            uploads = self.trainer.uplink_package
            self.handler.FedAvg(uploads)
            torch.save({"model": self.handler._model.state_dict(), "rounds": 0}, model_path)
            if metric:
                _,norm_noise_rates = self.collect_metrics(filepath=model_path, sampled_clients=sampled_clients)
                self.plot_noise_rate_comparison(norm_noise_rates, self.trainer.each_noise_ratio, save_path=os.path.join(self.out_path, f'noise_rate_comparison_round_{rnd}.png'))
            self.evaluate()
        torch.save({"model": self.handler._model.state_dict(), "rounds": self.args.begin}, model_path)

    def plot_noise_rate_comparison(self, norm_noise_rates, real_noise_rates, save_path=None):
        """
        对比归一化噪声率和真实噪声率，画出对比图，并记录MSE到txt文件。
        """
        import matplotlib.pyplot as plt

        x = np.arange(len(norm_noise_rates))
        plt.figure(figsize=(10, 5))
        plt.plot(x, norm_noise_rates, 'o-', label='Estimated Norm Noise Rate')
        plt.plot(x, real_noise_rates, 's-', label='Real Noise Rate')
        plt.xlabel('Client ID')
        plt.ylabel('Noise Rate')
        plt.title('Comparison of Estimated and Real Noise Rates')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

        # 计算MSE并保存
        mse = np.mean((np.array(norm_noise_rates) - np.array(real_noise_rates)) ** 2)
        mse_path = os.path.join(self.out_path, "noise_rate_mse.txt")
        with open(mse_path, "a") as f:
            f.write(f"{save_path if save_path else 'round'}: MSE={mse}\n")
    
    

    def collect_metrics(self, filepath='', sampled_clients=[]):
        if filepath != '':
            self.handler._model.load_state_dict(
                torch.load(filepath, map_location='cuda:' + str(torch.cuda.current_device()))['model']
            )
        broadcast = self.handler.downlink_package
    
        est_noise_rates = np.zeros(len(sampled_clients)).astype("float")
    
        # 1. 收集所有客户端的 metric_conf 和 est_noise_rate
        for idx, cid in enumerate(sampled_clients):
            _, est_noise_rate = self.trainer.class_metrics(cid, broadcast)
            est_noise_rates[idx] = est_noise_rate
    
        # 2. 计算 mean 和 std
        mean = np.mean(est_noise_rates)
        std = np.std(est_noise_rates) + 1e-8  # 防止除零
    
        # 3. 归一化
        norm_noise_rates = (est_noise_rates - mean) / std
        norm_noise_rates = np.clip(norm_noise_rates, 0, 1)
        
        return est_noise_rates, norm_noise_rates

    def load_model(self):
        criterion = 'LA' if self.args.warm_LA else 'ce'
        for m in ['ResNet18', 'ResNet20', 'ResNet34']:
            self.args.model = m
            self.nll_name = nllF.FedNLL_name(**vars(self.args))
            alg_name = make_alg_name(self.args)
            self.out_path = os.path.join(self.args.out_dir, self.nll_name, alg_name, self.exp_name)
            model_path = os.path.join(self.out_path, 'stage1_model.pth')
            checkpoint = torch.load(model_path)
            self.handler._model.load_state_dict(checkpoint['model'])
            self.handler.round = checkpoint['rounds'] - 1
            metrics_conf = self.collect_metrics(criterion=criterion, confidence=True, sampled_clients=range(self.args.num_clients))
            metrics_loss = self.collect_metrics(criterion=criterion, confidence=False, sampled_clients=range(self.args.num_clients))
            prclnt_mode = self.handler.clnt_loss_tab(metrics_conf, metrics_loss, range(self.args.num_clients))
            self.trainer.prclnt_mode = prclnt_mode

    
    
    def main(self):
        if os.path.exists(self.record_file):
            accs, _, _ = result_parser(self.record_file)
            if len(accs) >= self.args.com_round:
                print(f"Experiment done! Result saved in {self.record_file}!")
                return
        self.warm_up_stage()
        model_path = os.path.join(self.out_path, 'stage1_model.pth')
        for rnd in range(self.args.begin,self.args.end):
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package
            self.trainer.local_process_mask(broadcast, sampled_clients, rnd)
            uploads = self.trainer.uplink_package
            self.handler.FedAvg(uploads)
            # _,norm_noise_rates = self.collect_metrics(filepath=model_path, sampled_clients=sampled_clients)
            # self.trainer.est_noise_ratio = norm_noise_rates
            self.evaluate()
        torch.save({"model": self.handler._model.state_dict(), "rounds": self.args.begin}, model_path)
    



