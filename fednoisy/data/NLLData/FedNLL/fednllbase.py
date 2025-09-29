import random

import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from typing import Optional

from fedlab.utils.dataset.partition import DataPartitioner, VisionPartitioner

from fednoisy.data import (
    TRANSITION_MATRIX,   
    TRANSITION_MATRIX_SEM,
    TEST_TRANSFORM,
    TRAIN_TRANSFORM,
)
from fednoisy.data.NLLData.BaseNLL import NLLBase
from fednoisy.data.NLLData import functional as F
from fednoisy.data.NLLData.functional import NoisyDataset
import tqdm
import logging
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import BoundaryNorm
from rich.console import Console

console = Console()

class FedNLLScene(NLLBase):
    centralized = False

    def __init__(
        self,
        root_dir: str,
        out_dir: str, 
        min_require_size: int = 10,
        partition: str = "iid",
        num_clients: int=20,
        dir_alpha: float = 0.6,
        major_classes_num: int = -1,
        noise_mode: str = "norm",
        asym_rate: float = 1,
        norm_mean: float = 0.5,
        norm_std: float = 0.2,  
        partitioner: Optional[DataPartitioner] = VisionPartitioner,


        
    ) -> None:
        NLLBase.__init__(self, root_dir, noise_mode, out_dir)

        self.num_clients = num_clients
        if partition == "noniid-#label":
            # label-distribution-skew:quantity-based
            assert isinstance(major_classes_num, int), (
                f"'major_classes_num' should be integer, "
                f"not {type(major_classes_num)}."
            )
            assert major_classes_num > 0, f"'major_classes_num' should be positive."
            assert major_classes_num < self.num_classes, (
                f"'major_classes_num' for each client "
                f"should be less than number of total "
                f"classes {self.num_classes}."
            )

        elif partition in ["noniid-labeldir", "noniid-quantity"]:
            # label-distribution-skew(Dirichlet) and quantity-distribution-skew (Dirichlet)
            assert dir_alpha > 0, (
                f"Parameter 'dir_alpha' for Dirichlet distribution should be "
                f"positive."
            )

        elif partition == "iid" or partition == 'real':
            pass
        else:
            raise ValueError(
                f"Data partition only supports 'noniid-labeldir', 'noniid-quantity', 'iid', 'real'. "
                f"{partition} is not supported."
            )

        self.partition = partition

        
        self.clnt_noiseratio = {cid: 0.0 for cid in range(num_clients)}  # initial value
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.asym_rate = asym_rate
        self.min_require_size = min_require_size
        self.dir_alpha = dir_alpha
        self.major_classes_num = major_classes_num
        self.partitioner = partitioner
        self.noise_mode = noise_mode

    def create_nll_scene(self, seed: int = 0):
        
        self.setup_seed(seed)
        self.nll_scene_filename = f"{self}_seed_{seed}_setting.pt"
        self.nll_scene_file_path = os.path.join(self.out_dir, self.nll_scene_filename)
        self.nll_scene_folder = os.path.join(self.out_dir, f"{self}_seed_{seed}")
        client_dict = self._perform_partition()
        for cid in client_dict:
            client_dict[cid] = client_dict[cid].tolist()  # change array([...]) to [...]
        self.client_dict = client_dict
        self.data_dict = F.split_data(self.client_dict, self.train_data)
        self.labels_dict = F.split_data(client_dict, self.train_labels)
        self.noisy_labels_dict = self._gen_noisy_labels(client_dict)
        self.true_noise_ratio = F.cal_multiple_true_noisy_ratio(
            self.labels_dict, self.noisy_labels_dict
        )
        console.log(f"True noisy ratio is calculated: {self.true_noise_ratio}")

    def save_nll_scene(self):
        if os.path.exists(self.nll_scene_file_path):
            console.log(
                f"Federated noisy label learning scene {self}_seed_{self.seed} already generated, "
                f"loaded from {self.nll_scene_file_path}."
            )
        else:
            fednll_scene = {
                "dataset": self.dataset_name,
                "partition": self.partition,
                "num_clients": self.num_clients,
                "dir_alpha": self.dir_alpha,
                "noise_mode": self.noise_mode,
                "norm_mean": self.norm_mean,
                "norm_std": self.norm_std,
                "noise_ratio": self.clnt_noiseratio,
                "client_dict": self.client_dict,
                'clnt_mode': self.clnt_mode,
                "true_noise_ratio": self.true_noise_ratio,
                "noisy_labels": self.noisy_labels_dict,
            }

            torch.save(fednll_scene, self.nll_scene_file_path)
            console.log(
                f"Federated Noisy Label Learning scene saved to {self.nll_scene_file_path}, with keys "
                f"{list(fednll_scene.keys())}"
            )

            os.mkdir(self.nll_scene_folder)
            # train split save to local
            train_transform = TRAIN_TRANSFORM[self.dataset_name]
            for cid in range(self.num_clients):
                client_dataset = NoisyDataset(
                    data=self.data_dict[cid],
                    labels=self.labels_dict[cid],
                    noisy_labels=self.noisy_labels_dict[cid],
                    noise_mode=self.noise_mode,
                    train=True,
                    transform=train_transform,
                )
                path = os.path.join(self.nll_scene_folder, f"train-data{cid}.pkl")
                torch.save(client_dataset, path)
                console.log(f"Client {cid} local train set saved to {path}")

            # test split save to local
            test_transform = TEST_TRANSFORM[self.dataset_name]
            test_dataset = NoisyDataset(
                data=self.test_data,
                labels=self.test_labels,
                train=False,
                transform=test_transform,
            )
            path = os.path.join(self.nll_scene_folder, "test-data.pkl")
            torch.save(test_dataset, path)
            console.log(f"Test set saved to {path}")
            self.plot_clients_tsne(save_path=self.nll_scene_folder)



    def _perform_partition(self):
        if self.partition == "noniid-quantity":
            partition = "unbalance"
        else:
            partition = self.partition

        partitioner = self.partitioner(
            targets=np.array(self.train_labels),
            num_clients=self.num_clients,
            partition=partition,
            dir_alpha=self.dir_alpha,
            major_classes_num=self.major_classes_num,
            verbose=False,
            seed=self.seed,
        )

        return partitioner.client_dict

    def _generate_noise_norm(self):
        noisy_clients_num = int(self.num_clients)
        shuffled_idx = np.random.permutation(self.num_clients)
        noisy_clients_idx = shuffled_idx[:noisy_clients_num]
        noisy_clients = set(noisy_clients_idx)

        noise_ratios = np.random.normal(
            loc=self.norm_mean, scale=self.norm_std, size=noisy_clients_num
        )
        noise_ratios = np.clip(noise_ratios, 0, 1)
        self.clnt_noiseratio = {cid: noise_ratios[cid] for cid in range(self.num_clients)}
        asym_clients_num = int(noisy_clients_num * self.asym_rate)
        asym_clients_idx = set(shuffled_idx[:asym_clients_num])
        self.clnt_mode = {
            cid: (
                'asym' if cid in asym_clients_idx else 'sym'
            ) if cid in noisy_clients else 'clean'
            for cid in range(self.num_clients)
        }

    

    def _generate_all_client_noisy_labels(self):
        noisy_labels_dict = {}

        transition_matrix = (
            TRANSITION_MATRIX[self.dataset_name]
        )

        for cid in range(self.num_clients):
            clean_labels = self.labels_dict[cid]
            noisy_labels = F.generate_local_noisy_labels(
                clean_labels,
                noise_mode=self.noise_mode,
                bernoi_status=self.clnt_mode[cid],
                noise_ratio=self.clnt_noiseratio[cid],
                transition=transition_matrix,
                dataset=self.dataset_name,
            )
            noisy_labels_dict[cid] = noisy_labels

        return noisy_labels_dict

    def _gen_noisy_labels(self, client_dict):
        # if os.path.exists(self.nll_scene_file_path):
        #     console.log(
        #         f"Federated noisy label learning scene {self}_seed_{self.seed} already generated, "
        #         f"loaded from {self.nll_scene_file_path}."
        #     )
        #     entry = torch.load(self.nll_scene_file_path)
        #     noisy_labels_dict = entry["noisy_labels"]
        # else:
        if self.noise_mode == 'norm':
            self._generate_noise_norm()
        else:
            raise ValueError(f"Unsupported noise mode: {self.noise_mode}")

        noisy_labels_dict = self._generate_all_client_noisy_labels()
        console.log(f"Federated noisy label learning scene {self}_seed_{self.seed} generated.")

        return noisy_labels_dict

    @property
    def partition_setting(self):
        if self.partition == "noniid-#label":
            partition_param = f"{self.major_classes_num}"
        elif self.partition == "noniid-quantity":
            partition_param = f"{self.dir_alpha}"
        elif self.partition == "noniid-labeldir":
            partition_param = f"{self.dir_alpha:.2f}_{self.min_require_size}"
        else:
            # IID
            partition_param = ""
        return f"{self.num_clients}_{self.partition}_{partition_param}"

    @property
    def noise_setting(self):
        def fmt(k, v):
            return f"{k}_{v:.2f}" if isinstance(v, float) else f"{k}_{v}"

        if self.noise_mode == 'norm':
            info = {
                "mean": self.norm_mean,
                "std": self.norm_std,
            }

            noise_param = 'local_norm_'+"_".join([fmt(k, v) for k, v in info.items()])

        else:
            raise ValueError(f"Unsupported noise mode: {self.noise_mode}")

        return noise_param

    @property
    def setting(self):
        return f"{self.partition_setting}_{self.noise_setting}"
    

    def plot_clients_tsne(self, max_per_client=500, save_path=None):
        """
        分别对每个客户端的噪声标签数据做t-SNE降维并可视化，每个客户端单独一张图。
        :param max_per_client: 每个客户端最多采样多少数据点用于可视化
        :param save_path: 保存图片的文件夹路径，文件名自动加client id和.svg
        """
        # 获取所有客户端出现过的类别，保证颜色一致
        all_classes = np.unique(np.concatenate([np.unique(self.noisy_labels_dict[cid]) for cid in range(self.num_clients)]))
        n_classes = all_classes.max() + 1
        cmap = plt.get_cmap('tab20', n_classes)
        norm = BoundaryNorm(np.arange(-0.5, n_classes + 0.5), n_classes)

        if save_path is not None and not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        for cid in range(self.num_clients):
            data = self.data_dict[cid]
            labels = self.noisy_labels_dict[cid]
            idxs = np.random.choice(len(data), min(max_per_client, len(data)), replace=False)
            sampled_data = np.array(data)[idxs]
            sampled_labels = np.array(labels)[idxs]
            if len(sampled_data.shape) > 2:
                sampled_data = sampled_data.reshape(sampled_data.shape[0], -1)
            tsne = TSNE(n_components=2, random_state=42, init='pca')
            data_2d = tsne.fit_transform(sampled_data)
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(
                data_2d[:, 0], data_2d[:, 1], c=sampled_labels, cmap=cmap, norm=norm, alpha=0.7, s=12
            )
            cbar = plt.colorbar(scatter, ticks=all_classes)
            cbar.set_label("Noisy Label")
            plt.title(f"t-SNE of Noisy Data - Client {cid}")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.tight_layout()
            if save_path:
                out_path = os.path.join(save_path, f"client_{cid}_tsne.svg")
                plt.savefig(out_path, format='svg', dpi=300)
                plt.close()
            else:
                plt.show()
# ...existing code...