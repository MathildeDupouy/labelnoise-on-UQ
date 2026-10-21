import numpy as np
import torch


def add_noise(label, transition_matrix, categories, seed=42):
    """Change the label according to the probabilities in the label transition matrix.

    Args:
        label (int or torch.Tensor): the label that has to be modified
        transition_matrix (array): the transition matrix, each line must some to 1
        categories (list): the list of labels associated with each index in the transition  matrix.
                            The given label must be in the categories list
        seed (int): seed to generate the numpy generator from

    Raises:
        ValueError: If the label type is not recognized

    Returns:
        (int or torch.Tensor): The noisy label
    """
    rng = np.random.default_rng(seed)
    # hard label
    if type(label) == int:
        assert label in categories, f"Label {label} not in the given categories {categories}."
        label_idx = categories.index(label)
        probas = transition_matrix[label_idx, :]
        return rng.choice(categories, size=1, p=probas.numpy())[0]
    # soft label
    elif type(label) == torch.Tensor:
        true_hard = torch.argmax(label).item()
        assert true_hard in categories, f"Label {true_hard} not in the given categories {categories}."
        true_hard_idx = categories.index(true_hard)
        probas = transition_matrix[true_hard_idx, :]
        noisy_hard = rng.choice(categories, size=1, p=probas.numpy())[0]
        return torch.roll(label, shifts=noisy_hard - true_hard, dims=0)
    raise ValueError(f"Label type {type(label)} not recognized.")

def transition_matrix(noise_rate, categories):
    """Transition matrix as a torch Tensor for uniform noise with given noise_rate.

    Args:
        noise_rate (float): Probability of transition to another value
        categories (list): label names corresponding to each line and columns (order matters)

    Returns:
        torch.Tensor: transition matrix
    """
    transition_m = torch.zeros((len(categories), len(categories)))
    for class_i in range(len(categories)):
        for class_j in range(len(categories)):
            if class_i != class_j:
                transition_m[class_i, class_j] = noise_rate / (len(categories) - 1)
        # Risk of sum different of 1 with transition_m[class_i, class_j] = 1 - noise_rate
        transition_m[class_i, class_i] = 1. - torch.sum(transition_m[class_i, :])
    # Check transition_matrix property
    for i in range(len(categories)):
        np.testing.assert_almost_equal(float(torch.sum(transition_m[i, :])), 1., 6, f"Row {i} of the transition matrix does not sum to 1 ({torch.sum(transition_m[i, :])}).")
    
    return transition_m