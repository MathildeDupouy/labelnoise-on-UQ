from pathlib import Path

import numpy as np
import torch
from src.DataManipulation.noise import add_noise, transition_matrix
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from src.Utils.vocabulary import HARD_TARGET, INDEX, LABELS, NOISY, UNSUPERVISED


class MNISTDataset(Dataset):
    """Load MNIST dataset from a given data_path (and download it if needed) using torchvision.
    Subjects can be removed.
    Symmetric noise can be added to the labels, with a given rate. 
    A subset of class can be selected.
    """
    def __init__(self, data_path, train=True, val=False, val_ratio=0., label_keys=[HARD_TARGET], transform=None, noise_rate = 0, remove_class = [], seed = None):
        """

        Args:
            data_path (str or Path): Path to the MNIST dataset (will be downloaded at this path if it does not exist, see https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)
            train (bool, optional): If either train set (true) or test set (false) from torchvision dataset is used. Defaults to True.
            val (bool, optional): If the validation set should be used. Defaults to False.
            val_ratio (float, optional): Proportion of the training set to use for validation. Defaults to 0.
            label_keys (list, optional): Labels that will be used. Defaults to [HARD_TARGET].
            transform (torch.Transform object, optional): Pre-processing transformations to be applied on the images. Defaults to None.
            noise_rate (int, optional): The probability rate to change label, in range [0, 1]. Defaults to 0.
            remove_class (list, optional): Classes to remove from the dataset. Defaults to [] (no class removed).
            seed (int, optional): Seed for the label noise generator. If None, the seed will be randomly chosen. Defaults to None.
        """
        assert val_ratio >= 0 and val_ratio < 1, f"Validation ratio {val_ratio} must be in [0, 1)."
        assert (val and train) or not val, f"Validation set cannot be used if train is False. Got val={val} and train={train}."

        if transform is None:
            # Default transformations
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            list_transforms = []
            for t in transform:
                if t["name"] == "resize":
                    list_transforms.append(transforms.Resize(**t["args"]))
                else:
                    raise ValueError(f"Transform {t['name']} not recognized.")
            transform = transforms.Compose(list_transforms + [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        
        # Get dataset from torchvision
        self.data = datasets.MNIST(
            root=data_path,
            train=train,
            download=True,
            transform=transform
        )
    
        #----- Remove asked classes (inspired from https://discuss.pytorch.org/t/filtering-a-class-form-subset-of-fashonmnist/65162/5)
        # Get all targets
        targets = self.data.targets
        # Create a list of indices that will be masked
        target_indices = np.arange(len(targets))
        idx_to_remove = torch.zeros_like(targets).to(bool) 
        print(f"Removing classes {remove_class}")
        for classidx_to_remove in remove_class:
            idx_to_remove |= targets[target_indices]==classidx_to_remove
        data_idx = target_indices[~idx_to_remove]
        # Updating data
        self.data = Subset(self.data, data_idx)
        #----- Final categories list
        self.categories = sorted(list(set(np.unique(targets)).difference(set(remove_class))))
        
        # Handle validation set. Must be done after removing classes to have the correct validation size and categories in the validation set.
        if val_ratio > 0:
            # Split train dataset into train and validation sets
            train_data, val_data = torch.utils.data.random_split(self.data, [1 - val_ratio, val_ratio])
            if val:
                self.data = val_data
            else:
                self.data = train_data
        
        #----- Handle noise
        assert noise_rate >= 0 and noise_rate <= 1, f"Noise rate {noise_rate} must be in [0, 1]."
        self.noise_rate = noise_rate
        # Reproducibility
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.randint(0, 2**31 - 1) # Randomly choosing a seed (does not cover all the possible states of the rng generator)
        global_rng = np.random.default_rng(self.seed)
        # Deterministically generating index seeds
        self.index_seeds = global_rng.integers(low=0, high=2**31 - 1, size=len(self.data))
        # Create transition matrix
        self.label_transition_matrix = transition_matrix(self.noise_rate, self.categories)
        print("Transition matrix", self.label_transition_matrix)
        self.noisy_labels = [None] * len(self.data)
        
        #----- Labels
        self.label_keys = label_keys

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        
        item={INDEX: idx}
        
        # Building labels
        labels = {}
        if HARD_TARGET in self.label_keys:
            labels[HARD_TARGET] = label
            
        if UNSUPERVISED in self.label_keys:
            labels[UNSUPERVISED] = image
        item[LABELS] = {k: v for k, v in labels.items()} # Deep copy of labels
            
        # Adding noise
        if self.noise_rate > 0:
        # Compute noisy label if never computed (same noisy label during all training)
            true_label = labels[HARD_TARGET]
            if self.noisy_labels[idx] is None:
                self.noisy_labels[idx] = add_noise(label, self.label_transition_matrix, self.categories, seed=self.index_seeds[idx])
            noisy_label = self.noisy_labels[idx]
            labels[HARD_TARGET] = noisy_label
            item[NOISY] = (true_label, noisy_label)
        else:
            item[NOISY] = (-1, -1)
        
        return image, labels, item

if __name__ == "__main__":
    data_path = Path("/home/dupouy/Documents/data/")

    #----- Test classical dataset
    print("Testing classical dataset")
    # Create dataset instances
    noise_rate = 0.4
    train_dataset = MNISTDataset(data_path, train=True, val_ratio=0.1, noise_rate = noise_rate, remove_class=[0, 1, 2])
    val_dataset = MNISTDataset(data_path, train=True, val=True, val_ratio=0.1, noise_rate = noise_rate, remove_class=[0, 1, 2])
    test_dataset = MNISTDataset(data_path, train=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Example usage
    print(f"Train size: {len(train_dataset)}") # 60000 - |classes 0, 1, 2| = 41377 (42000 if balanced); 41377 * (1 - val_ratio) = 37239.3 ~ 37240 if val_ratio = 0.1
    print(f"Validation size: {len(val_dataset)}") # 41377 * val_ratio = 4137.7 ~ 4137
    print(f"Test size: {len(test_dataset)}") # 10000

    # for images, labels, item in train_loader:
    #     print(f"Batch size: {images.size(0)}, Image shape: {images.shape}, Labels: {labels} {type(labels)}") # 64, 1, 28, 28,
    #     break
    # noisy = 0
    # total = 0
    # for images, labels, item in train_dataset:
    #     total += 1
    #     noisy += 1 if item[NOISY][0] != item[NOISY][1] else 0
    # print(noisy, total, noisy/total, noise_rate)

    # # Check that item contains the true labels
    # noisy = 0
    # total = 0
    # for images, labels, item in train_dataset:
    #     total += 1
    #     noisy += 1 if item[LABELS][HARD_TARGET] != labels[HARD_TARGET] else 0
    # print(noisy, total, noisy/total, noise_rate)
    
    # # Check label noise and reproducibility
    # train_dataset2 = MNISTDataset(data_path,
    #                               train=True,
    #                               noise_rate = noise_rate,
    #                               remove_class=[0, 1, 2],
    #                               seed=train_dataset.seed
    #                               )
    # i_test = 50
    # print("seed", train_dataset.seed)
    # test_elem2_1 = train_dataset2[i_test - 1] # Accessing elements in a different order
    # test_elem2 = train_dataset2[i_test]
    # print("First element:")
    # print(type(test_elem2[0]), test_elem2[0].shape)
    # print(type(test_elem2[1]), test_elem2[1])
    # print("noisy ?", train_dataset[i_test][2][NOISY], train_dataset[i_test][2][NOISY])
    
    # count_identical = 0
    # count_noisy1 = 0
    # count_noisy2 = 0
    # for i in range(len(train_dataset)):
    #     x1, l1, info1 = train_dataset[i]
    #     x2, l2, info2 = train_dataset2[i]
        
    #     count_identical += 1. * (l1[HARD_TARGET] == l2[HARD_TARGET])
    #     if l1[HARD_TARGET] != l2[HARD_TARGET]:
    #         print("sample", i, l1[HARD_TARGET], l2[HARD_TARGET], info1[NOISY], info2[NOISY])
    #     count_noisy1 += 1. * (info1[NOISY][0] != info1[NOISY][1])
    #     count_noisy2 += 1. * (info2[NOISY][0] != info2[NOISY][1])
    # print(f"Noisy labels between train_dataset and train_dataset2 are {count_identical/len(train_dataset):.2%} identical ({count_identical}, {len(train_dataset)}, {len(train_dataset2)})")
    # print(f"Mode dataset 1 has {count_noisy1} noisy", count_noisy1/len(train_dataset))
    # print(f"Mode dataset 2 has {count_noisy2} noisy", count_noisy2 / len(train_dataset2))
    
    # print("seed", train_dataset.seed)
    # print("index seed", train_dataset.index_seeds)
    # print("index seed", train_dataset2.index_seeds)
    
    #------ Test with transformations
    # import matplotlib.pyplot as plt
    
    # print("Testing dataset with transformations")
    # transform1 = [{"name": "resize", "args": {"size": [100, 100], "interpolation": transforms.InterpolationMode.NEAREST}}]
    # train_dataset_big1 = MNISTDataset(data_path, train=True, noise_rate = noise_rate, transform=transform1)
    # transform2  = [{"name": "resize", "args": {"size": [100, 100], "interpolation": transforms.InterpolationMode.BILINEAR}}]
    # train_dataset_big2 = MNISTDataset(data_path, train=True, noise_rate = noise_rate, transform=transform2)
    
    # img1, labels1, item1 = train_dataset_big1[0]
    # img2, labels2, item2 = train_dataset_big2[0]
    # img_small, labels_small, item_small = train_dataset[0]
    # print(f"Image shape: {img1.shape}, Labels: {labels1}, Item: {item1}")
    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 3, 1)
    # plt.imshow(img_small.squeeze(), cmap='gray')
    # plt.title(f"Original (shape {img_small.shape})")
    # plt.subplot(1, 3, 2)
    # plt.imshow(img1.squeeze(), cmap='gray')
    # plt.title(f"Resized (shape {img1.shape})")
    # plt.subplot(1, 3, 2)
    # plt.imshow(img1.squeeze(), cmap='gray')
    # plt.title(f"Resized NEAREST (shape {img1.shape})")
    # plt.subplot(1, 3, 3)
    # plt.imshow(img2.squeeze(), cmap='gray')
    # plt.title(f"Resized BILINEAR (shape {img2.shape})")
    # plt.show()
    