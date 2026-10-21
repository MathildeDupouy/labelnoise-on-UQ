import numpy as np


def get_mean_std(dataset):
    samples = [sample[0] for sample in dataset]
    return np.mean(np.array(samples)), np.std(np.array(samples))

def get_class_dist(dataset, label_name):
    labels = [sample[1][label_name] for sample in dataset] # Potentially noisy labels, ok
    return np.unique(np.array(labels), return_counts=True)