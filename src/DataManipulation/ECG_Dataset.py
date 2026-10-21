import csv as csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from src.DataManipulation.constructors import (
    ECG_Audio_Constructor,
    ECG_Spectrogram_Constructor,
)
from src.DataManipulation.noise import add_noise, transition_matrix
from torch.utils.data import Dataset

from src.Utils.vocabulary import (
    HARD_TARGET,
    ID,
    INDEX,
    LABELS,
    NOISY,
    SIGNAL,
    UNSUPERVISED,
)


class ECG_Dataset(Dataset):
    """A dataset to store ECG from PTB csv files (pre-loaded from the Kaggle challenge https://www.kaggle.com/shayanfazeli/heartbeat), their labels and metadata.
    # A dictionary of samples is passed to the constructor, and the dataset is built from it.
    Subjects can be removed.
    Symmetric noise can be added to the labels, with a given rate. 
    """
    def __init__(self, 
                 data_list,
                 constructor,
                 constructor_args,
                 label_keys = [HARD_TARGET],
                 remove_subjects = [],
                 noise_rate = 0,
                 normalize = False,
                 normalize_mean_std = None,
                 seed = None
                 ) -> None:
        """
        Arguments:
        ----------
            data_list (List<dict>): Dictionaries for each sample, see ptb_split_train_test for dict structure.
            constructor (str in 'ECG_audio_constructor', 'ECG_spectro_constructor'): The constructor used to build the items (audio).
            constructor_args (dict): The arguments to pass to the constructor.
            label_keys (list, optional): Labels that will be used. Defaults to [HARD_TARGET].
            remove_subjects (list<list<int>>): The subjects (by their row number in each file) to remove from the dataset (example for PTB : ["abn-1"])
            noise_rate (float): The probability rate to change label, in range [0, 1]. Defaults to 0.
            normalize (bool, optional): if True, data will be normalized (minus mean divided by std). Mean and std values are computed or manually defined with normalize_mean_std argument. Defaults to False.
            normalize_mean_std (Tuple[float, float], optional): Mean and standard deviation to use for normalizetion. If None, they are computed on available data. Defaults to None.
            seed (int): Seed for the label noise generator. If None, the seed will be randomly chosen. Defaults to None.

        """
        super().__init__()

        print("======Creating dataset======")
        #----- Loading data
        self.remove_subjects = remove_subjects
        self.class_weights = defaultdict(lambda: 0)
        self.data = []
        for sample in data_list:
            assert SIGNAL in sample.keys(), f"Key {SIGNAL} not found in sample {sample}"
            assert LABELS in sample.keys(), f"Key {LABELS} not found in sample {sample}"
            assert ID in sample.keys(), f"Key {ID} not found in sample {sample}"
            if sample[ID] in self.remove_subjects:
                continue
            # Adding the sample to the list
            self.class_weights[sample[LABELS]] += 1
            self.data.append(sample)
        self.class_weights = dict(self.class_weights) # avoid keeping lambda if stored memory
        self.signal_length = len(sample[SIGNAL])
        
        #----- Constructor
        if constructor == "ECG_audio_constructor":
            self.constructor = ECG_Audio_Constructor(**constructor_args)
        elif constructor == "ECG_spectro_constructor":
            self.constructor = ECG_Spectrogram_Constructor(**constructor_args)
        else:
            raise ValueError(f"Constructor {constructor} not recognized (must be a string, one of ECG_spectro_constructor, ECG_audio_constructor).")

    
        #----- Noise
        assert noise_rate >= 0 and noise_rate <= 1, f"Noise rate {noise_rate} must be in [0, 1]."
        self.noise_rate = noise_rate
        # Reproducibility
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.randint(0, 2**32) # Randomly choosing a seed (does not cover all the possible states of the rng generator)
        global_rng = np.random.default_rng(self.seed)
        # Deterministically generating index seeds
        self.index_seeds = global_rng.integers(low=0, high=2**32, size=len(self.data))
        # Create transition matrix
        self.noisy_categories = list(self.class_weights.keys())
        self.noisy_categories.sort()
        self.label_transition_matrix = transition_matrix(self.noise_rate, self.noisy_categories)
        print("Transition matrix", self.label_transition_matrix)
        # Noisy label list (once a noisy label is fixed, keep the same during Dataset life)
        self.noisy_labels = [None] * len(self.data)
        
        #----- Labels
        self.label_keys = label_keys
        
        #----- Normalization
        self.normalize = normalize
        self.mean, self.std = 0., 1.# Normalization is initialized to identity
        if self.normalize:
            if normalize_mean_std is None:
                samples = torch.stack([self.constructor(item_data[SIGNAL]) for item_data in self.data])
                self.mean = torch.mean(samples)
                self.std = torch.std(samples)
            else:
                self.mean = normalize_mean_std[0]
                self.std = normalize_mean_std[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item_data = self.data[index]
        
        # Some metadata
        item = {k: v for k, v in item_data.items() if k not in [SIGNAL, LABELS]}
        item[INDEX] = index
        
        # Building x
        x = self.constructor(item_data[SIGNAL])
        x = (x - self.mean) / self.std # Identity if not self.normalize

        # Building labels
        labels = {}
        if HARD_TARGET in self.label_keys:
            labels[HARD_TARGET] = item_data[LABELS]

        if UNSUPERVISED in self.label_keys:
            labels[UNSUPERVISED] = x
            
        # Adding noise
        item[LABELS] = {k: v for k, v in labels.items()} # Storing true labels in item
        if self.noise_rate > 0:
            # Compute noisy label if never computed (same noisy label during all training)
            true_label = labels[HARD_TARGET]
            if self.noisy_labels[index] is None:
                self.noisy_labels[index] = self.add_noise(true_label, seed=self.index_seeds[index])
            noisy_label = self.noisy_labels[index]
            labels[HARD_TARGET] = noisy_label
            item[NOISY] = (true_label, noisy_label)
        else:
            item[NOISY] = (-1, -1)

        return x, labels, item
    
    def add_noise(self, label, seed):
        """Change the label according to the probabilities in the label transition matrix.

        Args:
            label (int or torch.Tensor): the label that has to be modified
            seed (int): seed for the noise generation

        Returns:
            (int or torch.Tensor): The noisy label
        """
        return add_noise(label, self.label_transition_matrix, self.noisy_categories, seed=seed)

        
def ptb_split_train_test(abnormal_path, normal_path, test_size=0.2):
    """Split the PTB dataset into train and test sets.
    Each sample is a dict with following structure:
    {    
    ID: string identifying the sample ("abn-i" or "n-i")
    SIGNAL: 1D tensor containing the ECG signal
    LABELS: int with the label
    }

    Args:
        abnormal_path (Path): Path to the abnormal data file.
        normal_path (Path): Path to the normal data file.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Two dictionaries containing the train and test data.
    """
    assert Path(abnormal_path).exists(), f"Path {abnormal_path} does not exist."
    assert Path(normal_path).exists(), f"Path {normal_path} does not exist."
    
    # Load data
    with open(abnormal_path, "r") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        abnormal_data = []
        for i, row in enumerate(csv_reader):
            # Adding the sample to the list
            abnormal_data.append({
                ID: f"abn-{i}",
                SIGNAL: torch.Tensor(row[:-1]),
                LABELS: int(float(row[-1])),
                })
    
    with open(normal_path, "r") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        normal_data = []
        for i, row in enumerate(csv_reader):
            # Adding the sample to the list
            normal_data.append({
                ID: f"n-{i}",
                SIGNAL: torch.Tensor(row[:-1]),
                LABELS: int(float(row[-1])),
                })

    # Split data
    n_abnormal = len(abnormal_data)
    n_normal = len(normal_data)

    n_test_abnormal = int(np.floor(n_abnormal * test_size))
    n_test_normal = int(np.floor(n_normal * test_size))
    print(f"Test set will contain {n_test_abnormal} abnormal and {n_test_normal} normal samples.")

    train_list = abnormal_data[:n_abnormal - n_test_abnormal] + normal_data[:n_normal - n_test_normal]

    test_list = abnormal_data[n_abnormal - n_test_abnormal:] + normal_data[n_normal - n_test_normal:]

    return train_list, test_list
        

###############################################################################
###############################################################################

if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt

    dataset_path = Path("/path/to/data/ECG_Heartbeat_Categorization_Dataset")
    
    #------- PTB test
    abnormal_path = dataset_path/"ptbdb_abnormal.csv"
    normal_path = dataset_path/"ptbdb_normal.csv"
    train_list, test_list = ptb_split_train_test(abnormal_path, normal_path, test_size=0.2)
    # Create dataset
    train_dataset = ECG_Dataset(
        train_list,
        constructor="ECG_audio_constructor",
        constructor_args={},
        noise_rate=0.2,
        seed=50
        )
    test_dataset = ECG_Dataset(
        test_list,
        constructor="ECG_audio_constructor",
        constructor_args={},
        noise_rate=0.0,
        )
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    for i in range(10):
        i_test = torch.randint(0, len(train_dataset), (1,)).item()
        x, label, item = train_dataset[i_test]
        print(item)
        print("First element:", len(x), type(x))
        plt.figure()
        plt.plot(x.squeeze())
        plt.show()
    print("Class weights:", train_dataset.class_weights)
    print("Noisy categories:", train_dataset.noisy_categories)
    print("Class weights:", test_dataset.class_weights)
    print("Noisy categories:", test_dataset.noisy_categories)
    
    # Test noise
    nb_noisy = 0
    for i in range(len(train_dataset)):
        x, labels, info = train_dataset[i]

        if type(info[NOISY][0]) == torch.Tensor:
            if not torch.equal(torch.Tensor(info[NOISY][0]), torch.Tensor(info[NOISY][1])):
                nb_noisy += 1
        else:
            if info[NOISY][0] != info[NOISY][1]:
                nb_noisy += 1
    print("Transition matrix", train_dataset.label_transition_matrix)

    print(f"Nb noisy: {nb_noisy} over {len(train_dataset)}")
    print(f"Observed noise ratio: {nb_noisy/len(train_dataset):.2f}")
    
    # Test reproducibility
    train_dataset2 = ECG_Dataset(
        train_list,
        constructor="ECG_audio_constructor",
        constructor_args={},
        noise_rate=0.2,
        seed=train_dataset.seed
        )
    print("seed", train_dataset.seed)
    test_elem2_1 = train_dataset2[i_test - 1] # Accessing elements in a different order
    test_elem2 = train_dataset2[i_test]
    print("First element:")
    print(type(test_elem2[0]), test_elem2[0].shape)
    print(type(test_elem2[1]), test_elem2[1])
    print("noisy ?", train_dataset[i_test][2][NOISY], train_dataset[i_test][2][NOISY])
    
    count_identical = 0
    count_noisy1 = 0
    count_noisy2 = 0
    for i in range(len(train_dataset)):
        x1, l1, info1 = train_dataset[i]
        x2, l2, info2 = train_dataset2[i]
        
        count_identical += 1. * (l1[HARD_TARGET] == l2[HARD_TARGET])
        if l1[HARD_TARGET] != l2[HARD_TARGET]:
            print("sample", i, l1[HARD_TARGET], l2[HARD_TARGET], info1[NOISY], info2[NOISY])
        count_noisy1 += 1. * (info1[NOISY][0] != info1[NOISY][1])
        count_noisy2 += 1. * (info2[NOISY][0] != info2[NOISY][1])
    print(f"Noisy labels between train_dataset and train_dataset2 are {count_identical/len(train_dataset):.2%} identical ({count_identical}, {len(train_dataset)}, {len(train_dataset2)})")
    print(f"Mode dataset 1 has {count_noisy1} noisy")
    print(f"Mode dataset 2 has {count_noisy2} noisy")
    
    print("seed", train_dataset.seed)
    print("index seed", train_dataset.index_seeds)
    print("index seed", train_dataset2.index_seeds)