import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from src.DataManipulation.constructors import (
    EEG_Audio_Constructor,
    EEG_Spectrogram_Constructor,
)
from src.DataManipulation.noise import add_noise, transition_matrix
from torch.utils.data import Dataset

from src.Utils.vocabulary import (
    BINARY_TARGET,
    ID,
    INDEX,
    LABELS,
    MULTI_TARGET,
    NOISY,
    SIGNAL,
    UNSUPERVISED,
    index2multitarget_ESR,
)


class EEG_Dataset(Dataset):
    """A dataset to store EEG from Eplileptic seizure recognition (ESR) csv file (pre-loaded from the Kaggle challenge https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition), their labels and metadata.
    # A dictionary of samples is passed to the constructor, and the dataset is built from it.
    Subjects can be removed.
    Symmetric noise can be added to the labels, with a given rate. 
    """
    def __init__(self,
                 data_list,
                 constructor,
                 constructor_args,
                 label_keys = [BINARY_TARGET, MULTI_TARGET],
                 remove_subjects = [],
                 noise_rate = 0,
                 noise_type = None,
                 normalize = False,
                 normalize_mean_std = None,
                 seed = None
                 ) -> None:
        """
        Signal length is computed on the last sample of the list (it is assumed that all signals have the same length)

        Args:
            data_list (List<dict>): Dictionaries for each sample, see ptb_split_train_test for dict structure.
            constructor (str in 'EEG_audio_constructor'): The constructor used to build the items (audio).
            constructor_args (dict): The arguments to pass to the constructor.
            label_keys (list, optional): Labels that will be used. Defaults to [BINARY_TARGET, MULTI_TARGET].
            remove_subjects (list, optional): Subjects that will be removed. Defaults to [].
            noise_rate (int, optional): he probability rate to change label, in range [0, 1].. Defaults to 0.
            noise_type (str): The type of noise to add to the dataset (None, BINARY_TARGET, MULTI_TARGET)
            normalize (bool, optional): if True, data will be normalized (minus mean divided by std). Mean and std values are computed or manually defined with normalize_mean_std argument. Defaults to False.
            normalize_mean_std (Tuple[float, float], optional): Mean and standard deviation to use for normalization. If None, they are computed on available data. Defaults to None.
            seed (int): Seed for the noise generator. If None, the seed will be randomly chosen. Defaults to None.
        """
        super().__init__()

        print("======Creating dataset======")
        #----- Loading data
        self.remove_subjects = remove_subjects
        self.data = []
        self.class_weights = defaultdict(lambda: 0) # Class weights for the multi target
        for sample in data_list:
            assert SIGNAL in sample.keys(), f"Key {SIGNAL} not found in sample {sample}"
            assert LABELS in sample.keys(), f"Key {LABELS} not found in sample {sample}"
            assert ID in sample.keys(), f"Key {ID} not found in sample {sample}"
            # Removing asked subjects
            if sample[ID] in self.remove_subjects:
                continue
            # Adding the sample to the list
            self.class_weights[sample[LABELS]] += 1
            self.data.append(sample)
        self.class_weights = dict(self.class_weights) # "dict" to avoid keeping lambda if stored memory
        self.signal_length = len(sample[SIGNAL])
        
        #----- Constructor
        if constructor == "EEG_audio_constructor":
            self.constructor = EEG_Audio_Constructor(**constructor_args)
        elif constructor == "EEG_spectro_constructor":
            self.constructor = EEG_Spectrogram_Constructor(**constructor_args)
        else:
            raise ValueError(f"Constructor {constructor} not recognized (must be a string, one of EEG_sticker_constructor, EEG_audio_constructor).")

        #----- Noise
        assert noise_rate >= 0 and noise_rate <= 1, f"Noise rate {noise_rate} must be in [0, 1]."
        assert noise_type in [None,BINARY_TARGET, MULTI_TARGET], f"Noise type {noise_type} not recognized (must be one of binary target, multi target)."
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        # Reproducibility
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.randint(0, 2**31 - 1) # Randomly choosing a seed (does not cover all the possible states of the rng generator)
        global_rng = np.random.default_rng(self.seed)
        # Deterministically generating index seeds
        self.index_seeds = global_rng.integers(low=0, high=2**32, size=len(self.data))
        # Build transition matrix
        self.noisy_categories = index2multitarget_ESR
        self.noisy_categories = [k for k in self.noisy_categories.keys() if k in self.class_weights.keys()] # remove classes not present
        if self.noise_type == BINARY_TARGET :
            bin_noisy = [0] if 0 in self.noisy_categories else []
            bin_noisy += [1] if (1 in self.noisy_categories or 2 in self.noisy_categories or 3 in self.noisy_categories or 4 in self.noisy_categories) else []
            self.noisy_categories = bin_noisy
        self.label_transition_matrix = transition_matrix(self.noise_rate, self.noisy_categories)
        print("Transition matrix", self.label_transition_matrix)
        self.noisy_labels = [None] * len(self.data)

        #----- Labels
        self.label_keys = label_keys
        
        #----- Get constructed samples, so that computation is only done once (no fft computed at each call of __getitem__)
        self.constructed_samples = [self.constructor(item_data[SIGNAL]) for item_data in self.data]

        #----- Normalization
        self.normalize = normalize
        self.mean, self.std = 0., 1.# Normalization is initialized to identity
        if self.normalize:
            if normalize_mean_std is None:
                samples = torch.stack(self.constructed_samples)
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
        x = self.constructed_samples[index]
        x = (x - self.mean) / self.std # Identity if not self.normalize

        # Building labels
        labels = {}
        if MULTI_TARGET in self.label_keys:
            labels[MULTI_TARGET] = item_data[LABELS]
        
        if BINARY_TARGET in self.label_keys:
            labels[BINARY_TARGET] = 0 if item_data[LABELS] == 0 else 1

        if UNSUPERVISED in self.label_keys:
            labels[UNSUPERVISED] = x
            
        # Adding noise
        item[LABELS] = {k: v for k, v in labels.items()} # Storing true labels in item
        if self.noise_rate > 0:
            # Compute noisy label if never computed (same noisy label during all training)
            true_label = labels[self.noise_type]
            if self.noisy_labels[index] is None:
                self.noisy_labels[index] = self.add_noise(true_label, seed=self.index_seeds[index])
            noisy_label = self.noisy_labels[index]
            labels[self.noise_type] = noisy_label
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
        
def esr_split_train_test(csv_path, test_size=0.2):
    """
    Opens a csv file from the Epileptic Seizure Recognition Dataset.
    The dataset is from : https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition
    and originally from : https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition
        
    The CSV is supposed to have the format given by the Kaggle dataset, i.e. line descriptor in the first column with the format X[i].V[i].[subid]
    and the label (1, 2, 3, 4 or 5) in the last column --> WARNING: converted to 0, 1, 2, 3, 4
    Each sample is a dict with following structure:
    {    
    ID: string identifying the sample ("abn-i" or "n-i")
    SIGNAL: 1D tensor containing the ECG signal
    LABELS: int with the label
    }
    
    Train/test separation is deterministically done at the line level, 
    so make sure that the classes are shuffled in the csv for same balance across sets
    
    Args:
        csv_path (str or Path): Path to the csv file of the Epileptic Seizure Recognition dataset.
        test_size (float): Proportion (value between 0 and 1) of the dataset to include in the test split.

    Returns:
        tuple: Two dictionaries containing the train and test data.
    """
    csv_path = Path(csv_path)
    sample_name_column = 0
    label_column = -1
    assert csv_path.exists(), f"[esr_split_train_test] CSV path {csv_path} not found"
    
    samples_per_class = defaultdict(lambda: 0)
    data_list = []
    
    with open(csv_path, "r", newline='') as csvfile:
        # CSV reader
        csv_reader = csv.reader(csvfile, delimiter=',')
        for i_row, row in enumerate(csv_reader):
            if i_row == 0:
                continue
            # Subject ID
            subject_id = row[sample_name_column].split('.')[-1]
            if 'V' in subject_id: # Some errors in csv from Kaggle
                subject_id = subject_id.replace('V1', 'V1.-0') # Subjects with v1... name are converted to negative ids
                subject_id = subject_id.split('.')[-1]
            # Label
            label_id = int(row[label_column]) - 1 # Initial id in 1...5, new id in 0...4
            samples_per_class[label_id] += 1
            # Signal
            signal = [float(si) for si in row[1:-1]]
            
            data_list.append({
                ID: subject_id,
                SIGNAL: torch.Tensor(signal),
                LABELS: label_id
            })

    print(f"ESR data has been loaded with classes {samples_per_class}")
    
    n_samples = len(data_list)
    n_train = round(n_samples * (1 - test_size))
    train_list = data_list[:n_train]
    test_list = data_list[n_train:]
    print(f"Train set will contain {n_train} ({n_train/n_samples:.2%}) samples and test set will contain {n_samples-n_train} ({(n_samples-n_train)/n_samples:.2%}) samples")
    return train_list, test_list

###############################################################################
###############################################################################

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # dataset_path = Path("/path/to/data/EEG_Epileptic_Seizure_Recognition")
    dataset_path =  Path(r"data\EEG_Epileptic_Seizure_Recognition")
    #------- ESR test
    csv_path = dataset_path/"Epileptic_Seizure_Recognition.csv"
    train_list, test_list = esr_split_train_test(csv_path, test_size=0.2)
    #=== 1. Create audio dataset
    print(f"=== 1. Create audio train and test dataset ===")
    train_dataset = EEG_Dataset(
        train_list,
        constructor="EEG_audio_constructor",
        label_keys = [MULTI_TARGET],
        constructor_args={},
        noise_rate=0.2,
        noise_type=MULTI_TARGET,
        normalize = True
        )
    test_dataset = EEG_Dataset(
        test_list,
        constructor="EEG_audio_constructor",
        label_keys=[MULTI_TARGET],
        constructor_args={},
        noise_rate=0.0,
        noise_type=None,
        normalize = True
        )
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(train_dataset.mean, train_dataset.std)
    print(test_dataset.mean, test_dataset.std)
    for i in range(10):
        i_test = torch.randint(0, len(train_dataset), (1,)).item()
        x, label, item = train_dataset[i_test]
        print(item)
        print("First element:", len(x), type(x))
        print(x.min(), x.max(), x.mean(), x.std())
        plt.figure()
        plt.plot(x.squeeze())
        plt.show()

    #= 1.1 Control represented subjects
    print(f"= 1.1 Control represented subjects ===")
    sub_ids = []
    for i in range(len(train_dataset)):
        sub_ids.append(train_dataset[i][2][ID])
    print(len(set(sub_ids)), max(sub_ids), min(sub_ids))
    sub_ids = []
    for i in range(len(test_dataset)):
        sub_ids.append(test_dataset[i][2][ID])
    print(len(set(sub_ids)), max(sub_ids), min(sub_ids))
    print('-01' in sub_ids, '-04' in sub_ids, '-03' in sub_ids, '-0' in sub_ids, '-02' in sub_ids)
    
    #= 1.2 Control represented classes
    print(f"= 1.2 Control represented classes ===")
    print("Class weights:", train_dataset.class_weights)
    print("Noisy categories:", train_dataset.noisy_categories)
    print("Class weights:", test_dataset.class_weights)
    print("Noisy categories:", test_dataset.noisy_categories)
    
    #= 1.3 Test label noise
    print(f"= 1.3 Test label noise ===")
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
    
    #= 1.4 Test reproducibility label noise
    print(f"= 1.4 Test reproducibility label noise ===")
    train_dataset2 = EEG_Dataset(
        train_list,
        constructor="EEG_audio_constructor",
        label_keys = [MULTI_TARGET],
        constructor_args={},
        noise_rate=0.2,
        noise_type=MULTI_TARGET,
        normalize = True,
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
        
        count_identical += 1. * (l1[train_dataset.noise_type] == l2[train_dataset.noise_type])
        if l1[train_dataset.noise_type] != l2[train_dataset.noise_type]:
            print("sample", i, l1[train_dataset.noise_type], l2[train_dataset.noise_type], info1[NOISY], info2[NOISY])
        count_noisy1 += 1. * (info1[NOISY][0] != info1[NOISY][1])
        count_noisy2 += 1. * (info2[NOISY][0] != info2[NOISY][1])
    print(f"Noisy labels between train_dataset and train_dataset2 are {count_identical/len(train_dataset):.2%} identical ({count_identical}, {len(train_dataset)}, {len(train_dataset2)})")
    print(f"Mode dataset 1 has {count_noisy1} noisy")
    print(f"Mode dataset 2 has {count_noisy2} noisy")
    
    print("seed", train_dataset.seed)
    print("index seed", train_dataset.index_seeds)
    print("index seed", train_dataset2.index_seeds)
    
    #=== 2. Create spectrogram dataset
    print(f"=== 2. Create spectrogram dataset ===")
    train_dataset_spectro = EEG_Dataset(
        train_list,
        constructor="EEG_spectro_constructor",
        label_keys = [MULTI_TARGET],
        constructor_args={
            "n_fft": 32,
            "n_overlap": 4,
            "window": "blackman",
            "sample_rate": 178
        },
        noise_rate=0.2,
        noise_type=MULTI_TARGET,
        normalize = True
        )
    test_dataset_spectro = EEG_Dataset(
        test_list,
        constructor="EEG_spectro_constructor",
        label_keys=[MULTI_TARGET],
        constructor_args={
            "n_fft": 32,
            "n_overlap": 4,
            "window": "blackman",
            "sample_rate": 178
        },
        noise_rate=0.0,
        noise_type=None,
        normalize = True
        )
    print("Initial signal length", train_dataset_spectro.signal_length, test_dataset_spectro.signal_length)
    
    
    print("First element:")
    x, label, item = train_dataset_spectro[i_test]
    print("x", type(x), x.shape)
    print("label", type(label), label)
    print("noisy ?", train_dataset_spectro[i_test][2][NOISY])
    
    for i in range(10):
        i_test = torch.randint(0, len(test_dataset_spectro), (1,)).item()
        x, label, item = test_dataset_spectro[i_test]
        x_signal, label_signal, item_signal = test_dataset[i_test]
        print(item)
        print("First element:", len(x), type(x))
        print(x.min(), x.max(), x.mean(), x.std())
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(x.squeeze())
        plt.title(f"Spectrogram (label {label[MULTI_TARGET]})")
        plt.subplot(1, 2, 2)
        plt.plot(x_signal.squeeze())
        plt.title(f"Signal (label {label_signal[MULTI_TARGET]})")
        plt.show()
    
        