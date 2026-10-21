
# This script is inspired from the code provided with the article "Towards Uncertainty Quantification in Medical Image Classification", Lohr et al., AIME 2024,
# see the reference at https://github.com/timoverse/uq-in-mi/blob/main/uncertainty.py
import numpy as np
from scipy.stats import entropy

# variable probs with shape [batch_size, num_classes, num_ensemble_members]

def total_uncertainty_entropy(probs):
    """For each batch sample, compute the mean probability prediction vector across ensemble members,
    then compute the entropy of this mean vector.
    The entropy is normalized by the log2 of the number of classes (maximum entropy, because entropy of a uniform distribution) to get a value between 0 and 1.

    Args:
        probs (torch.Tensor): Predicted probabilities of shape [batch_size, num_classes, num_ensemble_members]

    Returns:
        array_like: total uncertainty entropy of shape [batch_size]
    """
    # print("Probs shape", probs.shape)
    # print(probs)
    t_u = entropy(np.mean(probs, axis=2), axis=1, base=2) / np.log2(probs.shape[1])
    return t_u


def epistemic_uncertainty_entropy(probs):
    """For each batch sample, compute the Kullback-Leibler divergence between the predicted probabilities and the mean predicted probabilities across ensemble members.
    This evaluate the "deviations" from each ensemble prediction to the mean prediction. This is then averaged across classes to get a single value per sample.
    The entropy is normalized by the log2 of the number of classes (maximum entropy, because entropy of a uniform distribution) to get a value between 0 and 1.

    Args:
        probs (torch.Tensor): Predicted probabilities of shape [batch_size, num_classes, num_ensemble_members]

    Returns:
        array_like: epistemic uncertainty entropy of shape [batch_size]
    """
    mean_probs = np.mean(probs, axis=2)
    mean_probs = np.repeat(np.expand_dims(mean_probs, 2), repeats=probs.shape[2], axis=2)
    e_u = entropy(probs, mean_probs, axis=1, base=2) / np.log2(probs.shape[1])
    e_u = np.mean(e_u, axis=1)
    return e_u


def aleatoric_uncertainty_entropy(probs):
    """_For each batch sample, compute the entropy of each class then average by member of the ensemble.
    It evaluates if the predictions are uncertain or not, by computing the entropy of the predicted probabilities.
    The entropy is normalized by the log2 of the number of classes (maximum entropy, because entropy of a uniform distribution) to get a value between 0 and 1.

    Args:
        probs (torch.Tensor): Predicted probabilities of shape [batch_size, num_classes, num_ensemble_members]

    Returns:
        array_like: aleatoric uncertainty entropy of shape [batch_size]
    """
    a_u = entropy(probs, axis=1, base=2) / np.log2(probs.shape[1])
    a_u = np.mean(a_u, axis=1)
    return a_u


def uncertainty_1vsall_entropy(probs, kind='tu'):
    """For each batch sample, compute the ..._uncertainty_entrop (where ... depends of which kind of uncertainty is asked) of probs and 1-probs then normalize by the corresponding factor.

    Args:
        probs (torch.Tensor): Predicted probabilities of shape [batch_size, num_classes, num_ensemble_members]

    Returns:
        array_like: uncertainty 1vsall entropy of shape [batch_size]
    """
    l_fun = lambda y: 0 if y == 0 else -np.log2(y)
    l_fun = np.vectorize(l_fun)
    k = probs.shape[1]
    fact = np.log2(k) + (k - 1) * np.log2(k / (k - 1))

    # probs shape (N,K,M)
    ps = np.stack([probs, 1 - probs])
    # stack shape (2,N,K,M)

    if kind == 'tu':
        out = np.sum(ps.mean(-1) * l_fun(ps.mean(-1)), 2).sum(0)
    elif kind == 'au':
        out = np.sum(ps * l_fun(ps), 2).mean(-1).sum(0)
    elif kind == 'eu':
        out = np.sum((ps * (l_fun(ps.mean(-1, keepdims=True)) - l_fun(ps))), 2).mean(-1).sum(0)
    return out / fact


def total_uncertainty_variance(probs):
    """For each batch sample, same computation as total_uncertainty_entropy, but instead of computing the entropy, 
    we compute h(1-h).

    Args:
        probs (torch.Tensor): Predicted probabilities of shape [batch_size, num_classes, num_ensemble_members]

    Returns:
        array_like: total uncertainty entropy of shape [batch_size]
    """
    mean = np.mean(probs, axis=2)
    t_u = np.sum(mean * (1 - mean), axis=1)
    return t_u


def aleatoric_uncertainty_variance(probs):
    """For each batch sample, same computation as aleatoric_uncertainty_entropy, but instead of computing the entropy, 
    we compute h(1-h).

    Args:
        probs (torch.Tensor): Predicted probabilities of shape [batch_size, num_classes, num_ensemble_members]

    Returns:
        array_like: aleatoric uncertainty entropy of shape [batch_size]
    """
    a_u = np.mean(np.sum(probs * (1 - probs), axis=1), axis=1)
    return a_u


def epistemic_uncertainty_variance(probs):
    """same computation as epistemic_uncertainty_entropy, but instead of computing the entropy, 
    we compute h(1-h).

    Args:
        probs (torch.Tensor): Predicted probabilities of shape [batch_size, num_classes, num_ensemble_members]

    Returns:
        array_like: epistemic uncertainty entropy of shape [batch_size]
    """
    mean = np.mean(probs, axis=2, keepdims=True)
    e_u = np.mean(np.sum(probs * (probs - mean), axis=1), axis=1)
    return e_u


def total_uncertainty_variance_label(probs):
    """Same computation as total_uncertainty_variance, but returns an uncertainty for each class (no sum over classes).

    Args:
        probs (torch.Tensor): Predicted probabilities of shape [batch_size, num_classes, num_ensemble_members]

    Returns:
        array_like: total uncertainty entropy of shape [batch_size, num_classes]
    """
    mean = np.mean(probs, axis=2)
    t_u = mean * (1 - mean)
    return t_u


def aleatoric_uncertainty_variance_label(probs):
    """Same computation as total_uncertainty_variance, but returns an uncertainty for each class (no sum over classes).

    Args:
        probs (torch.Tensor): Predicted probabilities of shape [batch_size, num_classes, num_ensemble_members]

    Returns:
        array_like: aleatoric uncertainty entropy of shape [batch_size, num_classes]
    """
    a_u = np.mean(probs * (1 - probs), axis=2)
    return a_u


def epistemic_uncertainty_variance_label(probs):
    """Same computation as total_uncertainty_variance, but returns an uncertainty for each class (no sum over classes).

    Args:
        probs (torch.Tensor): Predicted probabilities of shape [batch_size, num_classes, num_ensemble_members]

    Returns:
        array_like: epistemic uncertainty entropy of shape [batch_size, num_classes]
    """
    mean = np.mean(probs, axis=2, keepdims=True)
    e_u = np.mean(probs * (probs - mean), axis=2)
    return e_u

if __name__ == "__main__":
    # Test the functions
    probs = np.random.rand(3, 3, 5)
    print("Total uncertainty entropy:", total_uncertainty_entropy(probs)) #.shape)
    print("Aleatoric uncertainty entropy:", aleatoric_uncertainty_entropy(probs)) #.shape)
    print("Epistemic uncertainty entropy:", epistemic_uncertainty_entropy(probs)) #.shape)
    print("Total uncertainty variance:", total_uncertainty_variance(probs)) #.shape)
    print("Aleatoric uncertainty variance:", aleatoric_uncertainty_variance(probs)) #.shape)
    print("Epistemic uncertainty variance:", epistemic_uncertainty_variance(probs)) #.shape)
    print("Total uncertainty variance label:", total_uncertainty_variance_label(probs)) #.shape)
    print("Aleatoric uncertainty variance label:", aleatoric_uncertainty_variance_label(probs)) #.shape)
    print("Epistemic uncertainty variance label:", epistemic_uncertainty_variance_label(probs)) #.shape)
    print("Total uncertainty 1vsall entropy:", uncertainty_1vsall_entropy(probs, kind='tu')) #.shape)
    print("Aleatoric uncertainty 1vsall entropy:", uncertainty_1vsall_entropy(probs, kind='au')) #.shape)
    print("Epistemic uncertainty 1vsall entropy:", uncertainty_1vsall_entropy(probs, kind='eu')) #.shape)