"""This code aim at taking a train list from an unbalanced dataset and subsampling it to make it balanced.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.Utils.vocabulary import LABELS


def balance_by_subsampling(train_list, seed=42):
    rng = np.random.default_rng(seed=seed)
    
    # Compute number of samples by class
    class_dict = defaultdict(lambda: 0)
    class_samples = defaultdict(lambda: [])
    for sample in train_list:
        label = sample[LABELS]
        class_dict[label] += 1
        class_samples[label].append(sample)
    # Draw only a part of the samples
    new_train_list = []
    target_nb = min(class_dict.values())
    for c in class_dict.keys():
        nb_c = class_dict[c]
        sub_indices = rng.choice(np.arange(nb_c), target_nb, replace=False) # Draw target_nb indices between 0 and nb_c
        new_train_list += [class_samples[c][i] for i in sub_indices] # Concatenate list for the class to the global list
        
    return new_train_list

def noise_rate_lim(class_numbers):
    """Compute the minimum noise rate such than, with uniform noise applied on the distribution of classes given, a class i has more samples annotated i from class j != i than from class i itself.

    Args:
        class_numbers (list<int>): list of the number of samples for each class
    
    Returns:
        (float) limit noise rate for this dataset
        (int) index of class i (that has less samples in its own class)
        (int) index of class j (that has more samples labelled i than class i itself)
    """
    K = len(class_numbers)
    eta_lim_min = 1
    classi, classj = None, None
    for i, ni in enumerate(class_numbers):
        for j, nj in enumerate(class_numbers):
            eta_lim = ni / (ni + nj / (K - 1))
            if eta_lim <= eta_lim_min:
                eta_lim_min = eta_lim
                classi, classj = i, j
    return eta_lim_min, classi, classj

if __name__ == "__main__":
    #=========== Test balance_by_subsampling ===========
    from src.DataManipulation.ECG_Dataset import ptb_split_train_test

    dataset_path = Path("/path/to/data/ECG_Heartbeat_Categorization_Dataset")
    #------- PTB test
    abnormal_path = dataset_path/"ptbdb_abnormal.csv"
    normal_path = dataset_path/"ptbdb_normal.csv"
    train_list, test_list = ptb_split_train_test(abnormal_path, normal_path, test_size=0.2)
    new_train_list = balance_by_subsampling(train_list, 50)
    
    # Test class sizes
    class_dict = defaultdict(lambda: 0)
    for sample in new_train_list:
        label = sample[LABELS]
        class_dict[label] += 1
        # print(sample[LABELS], sample["signal"].sum()) # To check if samples are similar
    
    for sample in test_list:
        label = sample[LABELS]
        class_dict[str(label) + "_test"] += 1
    
    print(class_dict)
    
    #=========== Test noise rate limit ===========
    # Toy
    class_numbers = [100, 200, 300]
    print(f"Class numbers: {class_numbers}")
    eta_lim, classi, classj = noise_rate_lim(class_numbers)
    print(f"Noise rate limit: {eta_lim}, class i: {classi}, class j: {classj}")
    
    # Real datasets
    # MNIST
    from src.DataManipulation.MNIST_Dataset import MNISTDataset

    from src.Utils.vocabulary import HARD_TARGET, LABELS, MULTI_TARGET
    print("-- MNIST creation")
    data_path = Path("/path/to/data/")
    train_dataset = MNISTDataset(data_path, train=True)
    samples_classes = np.array([sample[1][HARD_TARGET] for sample in train_dataset])
    class_numbers_mnist = [np.sum(samples_classes == i) for i in np.unique(samples_classes)]
    
    # PTB sub
    print("-- PTB sub creation")
    from src.DataManipulation.ECG_Dataset import ECG_Dataset, ptb_split_train_test
    train_list, test_list = ptb_split_train_test(
        "/path/to/data/ECG_Heartbeat_Categorization_Dataset/ptbdb_abnormal.csv", 
        "/path/to/data/ECG_Heartbeat_Categorization_Dataset/ptbdb_normal.csv",
        test_size=0.2
        )
    train_list = balance_by_subsampling(train_list, seed=42)
    train_dataset = ECG_Dataset(
        train_list,
        constructor="ECG_audio_constructor",
        constructor_args={},
        )
    samples_classes = np.array([sample[1][HARD_TARGET] for sample in train_dataset])
    print(samples_classes)
    class_numbers_ptb_sub = [np.sum(samples_classes == i) for i in np.unique(samples_classes)]

    # PTB
    print("-- PTB creation")
    from src.DataManipulation.ECG_Dataset import ECG_Dataset, ptb_split_train_test
    train_list, test_list = ptb_split_train_test(
        "/path/to/data/ECG_Heartbeat_Categorization_Dataset/ptbdb_abnormal.csv", 
        "/path/to/data/ECG_Heartbeat_Categorization_Dataset/ptbdb_normal.csv",
        test_size=0.2
        )
    train_dataset = ECG_Dataset(
        train_list,
        constructor="ECG_audio_constructor",
        constructor_args={},
        )
    samples_classes = np.array([sample[1][HARD_TARGET] for sample in train_dataset])
    print(samples_classes)
    class_numbers_ptb = [np.sum(samples_classes == i) for i in np.unique(samples_classes)]

    # ESR
    print("-- ESR creation")
    from src.DataManipulation.EEG_Dataset import EEG_Dataset, esr_split_train_test
    train_list, test_list = esr_split_train_test(
        "/path/to/data/EEG_Epileptic_Seizure_Recognition/Epileptic_Seizure_Recognition.csv",
        test_size=0.2
        )
    train_dataset = EEG_Dataset(
        train_list,
        constructor="EEG_audio_constructor",
        constructor_args={}
        )
    samples_classes = np.array([sample[1][MULTI_TARGET] for sample in train_dataset])
    class_numbers_esr = [np.sum(samples_classes == i) for i in np.unique(samples_classes)]
    
    for dataset_name, class_numbers in zip(["MNIST", "PTB sub", "PTB", "ESR"], [class_numbers_mnist, class_numbers_ptb_sub, class_numbers_ptb, class_numbers_esr]):
        eta_lim, classi, classj = noise_rate_lim(class_numbers)
        print(f"----------------------------")
        print(f"Dataset: {dataset_name}, Class numbers: {class_numbers}")
        print(f"Noise rate limit: {eta_lim}, class i: {classi}, class j: {classj}")
        print(f"Noise rate theo: {(len(class_numbers) - 1 )/ len(class_numbers)}")

