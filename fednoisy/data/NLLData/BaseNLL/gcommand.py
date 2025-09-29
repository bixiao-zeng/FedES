import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter

from typing import Dict, Tuple, List, Optional

from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
    TRANSITION_MATRIX,
    CIFAR10_TRANSITION_MATRIX,
)
from fednoisy import visual
from fednoisy.data.NLLData import functional as F
from fednoisy.data.NLLData.BaseNLL import NLLBase
from fednoisy.utils.misc import make_dirs
import logging
import pickle


class NLLGCOMMAND(NLLBase):
    """
    Read raw train & test data from root_dir/cifar-10-batches-py, reformat image data into HWC np.array format,
    and generate noisy labels for train data, save them into local files:

    - Train file cifar10-trainset.pt content:
        {
        data: np.array[...],  # np.array images in HWC format
        labels: List[int],  # list of labels, label is in range of [0,9]
        class_to_idx: {class_name: class_label, ...}  # a dictionary mapping class_name to 0-9 class label
        classes: List[str],  # class names for 0-9 classes
        }

    - Test file cifar10-testset.pt content:
        {
        data: np.array[...],  # np.array images in HWC format
        labels: List[int],  # list of labels, label is in range of [0,9]
        class_to_idx: {class_name: class_label, ...}  # a dictionary mapping class_name to 0-9 class label
        classes: List[str],  # class names for 0-9 classes
        }

    - Noisy labels *.json file content:
       {
       noisy_labels: List[int],  # list of noisy labels
       noise_mode: str,  # noisy mode, 'sym'/'asym'/'clean'
       noise_ratio: float,  # noise ratio to generate the noise
       true_noise_ratio: float  # true noise ratio calculated using noisy_labels and train_labels
       }


    Args:
        root_dir (str): Root directory with downloaded CIFAR10 raw data files.
        noise_mode (str): Noise mode for centralized CIFAR10. Only 'sym', 'asym' and 'clean' are supported.
        noise_ratio (float): Noise ratio that is in range of [0, 1].
        out_dir: str, Output directory to save processed trainset/testset and noisy label file.
        seed (int) Random seed for noisy label generation.

    """

    dataset_name = "gcommand"

    num_classes = CLASS_NUM[dataset_name]
    train_sample_num = TRAIN_SAMPLE_NUM[dataset_name]
    test_sample_num = TEST_SAMPLE_NUM[dataset_name]
    trainset_filename = f"{dataset_name}-trainset.pt"
    testset_filename = f"{dataset_name}-testset.pt"
    # centralized = True

    def __init__(self, root_dir: str, noise_mode: str, out_dir: str) -> None:
        NLLBase.__init__(self, root_dir, noise_mode, out_dir)

        # self._load_testset()
        # self._load_trainset()

    def _load_meta(self) -> None:
        # process meta dataset info
        meta_file_path = os.path.join(
            self.root_dir, self.base_folder, self.meta["filename"]
        )
        meta_info = F.unpickle(meta_file_path)
        self.classes = meta_info[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _load_testset(self, save: bool = True) -> None:
        """

        Args:
            save (bool): Whether to save into local cifar10-testset.pt file. Default as ``True``.

        Returns:

        """
        # load from processed file, or from raw test data file
        if os.path.exists(self.testset_path):
            entry = torch.load(self.testset_path)
            test_data = entry["data"]
            test_labels = entry["labels"]
        else:
            # process raw test dataset
            # read from file, reshape and transbose data
            file_name = self.test_list[0][0]
            file_path = os.path.join(self.root_dir, self.base_folder, file_name)
            entry = F.unpickle(file_path)
            test_data = entry["data"]
            test_data = test_data.reshape((self.test_sample_num, 3, 32, 32))
            test_data = test_data.transpose((0, 2, 3, 1))  # convert to HWC
            if "labels" in entry:
                test_labels = entry["labels"]
            else:
                test_labels = entry["fine_labels"]

        self.test_data = test_data
        self.test_labels = test_labels
        print(f"{self.dataset_name} testset is loaded.")
        if save is True:
            self.save_testset()


    def _load_trainset(self, save: bool = True) -> None:
        """

        Args:
            save (bool): Whether to save into local cifar10-trainset.pt file. Default as ``True``.

        Returns:

        """
        # load data and model
        logging.info('Processing the raw data')
        load_file_path = '/workspace/ZBX/FedAudio/data/g_command/federated_dataset_raw_mel_spec.p'
        dataset = pickle.load(open(load_file_path, "rb"))
        logging.info("dataset has been loaded from saved file")


        # load from processed file, or from raw train data file
        if os.path.exists(self.trainset_path):
            entry = torch.load(self.trainset_path)
            train_data = entry["data"]
            train_labels = entry["labels"]
        else:
            train_data = []
            train_labels = []
            # process raw train dataset
            # read from files, concatenate batches, reshape and transpose data
            for file_name, _ in self.train_list:
                file_path = os.path.join(self.root_dir, self.base_folder, file_name)
                entry = F.unpickle(file_path)
                train_data.append(entry["data"])
                if "labels" in entry:
                    train_labels.extend(entry["labels"])
                else:
                    train_labels.extend(entry["fine_labels"])
            train_data = np.concatenate(
                train_data
            )  # concatenate raw train data from batch files
            train_data = train_data.reshape((self.train_sample_num, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))  # convert to HWC

        self.train_data = train_data
        self.train_labels = train_labels
        print(f"{self.dataset_name} trainset is loaded.")
        if save is True:
            self.save_trainset()


