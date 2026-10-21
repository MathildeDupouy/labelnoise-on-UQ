import json
import random
from collections.abc import Mapping
from typing import LiteralString

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_centered_range(new_duration, old_width, old_duration):
    """
    Get a range of index as a duration to be the closest of new_duration
    while keeping a parity similar to old_width to keep the middle pixels in the middle
    """
    width_raw = old_width / old_duration * new_duration
    middle_index = int(old_width / 2)
    odd_old_width = old_width % 2
    # If rounding down is not even, rounding up
    new_width = int(width_raw) + 1 * (int(width_raw)%2 != odd_old_width)
    duration = new_width * old_duration / old_width # real vignette duration
    start = middle_index - (new_width // 2)
    end = middle_index + (new_width // 2) + odd_old_width
    return start, end, duration

def get_closest_multiple(n, x) -> int:
    """
    Get the closest integer of n that is a multiple of x.
    Arguments:
    ----------
        n: int
        number to approximate

        x: int
        divider of the result
    Returns:
    --------
        int: result
    """
    res = n + x / 2
    res = res - (res % x)
    return res

def update_dict(old_dict, new_dict) -> dict:
    """Update old_dict with new_dict, replacing values for existing key and adding new key, values pair from new_dict.
    Values are recursively updated if they are dict, and concatenated if they are list.
    Beware: old_dict is not only returned but also modified in place, so if you want to keep the original dict, make a copy before calling this function.
    Arguments:
    ----------
    old_dict: dict
        dict to update
    new_dict: dict
        dict to fill old_dict with

    Returns:
    --------
    dict: updated dict
    """
    for k, v in new_dict.items():
        if isinstance(v, Mapping):
            # print("key", k, "value is a dict")
            old_dict[k] = update_dict(old_dict.get(k, {}), v)
        elif isinstance(v, list) and k in old_dict.keys():
            # print("key", k, "value is a list")
            old_dict[k] += v
        else:
            # print("key", k, "value is created/replaced")
            old_dict[k] = v
    return old_dict

def update_saved_dict(file_path, new_dict) -> None:
    """
    Load a dict from a json file and replace values for existing key and add new key, values pair from new_dict
    """
    if file_path.exists():
        with file_path.open("r") as f:
            new_dict_old = json.load(f)
        new_dict = update_dict(new_dict_old, new_dict)
    with file_path.open("w") as f:
        json.dump(new_dict, f, indent=4)

def format_classification_report(d, target_names_dict=None) -> LiteralString:
    """Given a dictionary returned by sklearn.metrics.classification_report with output_dict = true,
       returns a formatted string to be printed in a readable way, with the same format at if output_dict was false.
       The input dict is expected to have the following format:
    {
    "class1": {"precision": p1, "recall": r1, "f1-score": f1_1, "support": s1},
    "class2": {"precision": p2, "recall": r2, "f1-score": f1_2, "support": s2},
    ...
    "accuracy": a,
    "macro avg": {"precision": p_macro, "recall": r_macro, "f1-score": f1_macro, "support": s_macro},
    "weighted avg": {"precision": p_weighted, "recall": r_weighted, "f1-score": f1_weighted, "support": s_weighted}
    }
    A target_names_dict can be provided to replace the class labels by more readable names in the output string. It should have the following format:
    {
        "class1": "Readable Name 1",
        "class2": "Readable Name 2",
        ...
    }

    Args:
        d (_type_): _description_
        target_names_dict (_type_, optional): _description_. Defaults to None.

    Returns:
        LiteralString: _description_
    """
    lines = []
    
    # Header
    lines.append(" " * 16 + "precision    recall  f1-score   support\n")
    
    # Class rows (exclude summary keys)
    for label, metrics in d.items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue
        
        if target_names_dict is not None:
            label = target_names_dict.get(label, label)
        lines.append(
            f"{label:>15} "
            f"{metrics['precision']:>9.2f} "
            f"{metrics['recall']:>9.2f} "
            f"{metrics['f1-score']:>9.2f} "
            f"{int(metrics['support']):>9}\n"
        )
    
    lines.append("\n")
    
    # Accuracy row
    accuracy = d["accuracy"]
    total_support = int(d["weighted avg"]["support"])
    lines.append(
        f"{'accuracy':>15} "
        f"{'':>9} {'':>9} "
        f"{accuracy:>9.2f} "
        f"{total_support:>9}\n"
    )
    
    # Macro avg
    macro = d["macro avg"]
    lines.append(
        f"{'macro avg':>15} "
        f"{macro['precision']:>9.2f} "
        f"{macro['recall']:>9.2f} "
        f"{macro['f1-score']:>9.2f} "
        f"{int(macro['support']):>9}\n"
    )
    
    # Weighted avg
    weighted = d["weighted avg"]
    lines.append(
        f"{'weighted avg':>15} "
        f"{weighted['precision']:>9.2f} "
        f"{weighted['recall']:>9.2f} "
        f"{weighted['f1-score']:>9.2f} "
        f"{int(weighted['support']):>9}\n"
    )
    
    return "".join(lines)

# Function to plot the signals
def plot_stereo_waveform(time_axis, audio_signal) -> None:
    # Plot the left and right channels
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, audio_signal[:, 0], label='Canal gauche', alpha=0.7)
    plt.plot(time_axis, audio_signal[:, 1], label='Canal droit', alpha=0.7)
    plt.xlabel('Temps [s]')
    plt.ylabel('Amplitude')
    # plt.xlim(0, 2)
    plt.title("Extrait d'un enregistrement Doppler")
    plt.legend()
    plt.grid(True)
    plt.show()

def zero_pad_audio(audio_signal, target_length) -> np.ndarray[float]:
    """
    Pad the audio signal with zeros to reach the target length.
    If the audio signal is longer than the target length, it is truncated on both sides to keep the middle part.
    Else, zeros are added at both ends of the signal to keep the middle part in the middle of the output.

    Arguments:
    ----------
    audio_signal (np.ndarray): The input audio signal, shape (n_channels, n_samples).
    target_length (int): The desired length of the output signal.

    Returns:
    --------
    np.ndarray: The zero-padded audio signal, shape (n_channels, target_length).
    """
    n_channels, n_samples = audio_signal.shape
    if n_samples >= target_length:
        # Truncate the signal to keep the middle part
        start = (n_samples - target_length) // 2
        end = start + target_length
        return audio_signal[:, start:end]
    # Pad the signal with zeros to keep the middle part in the middle 
    pad_size_start = (target_length - n_samples) // 2
    padding = (pad_size_start, target_length - n_samples - pad_size_start)
    return np.pad(audio_signal, pad_width=((0, 0), padding), mode='constant', constant_values=0)

def set_seed(seed) -> None:
    """Set the seed for the random number generators of random, numpyr, torch and torch.cuda for reproducibility.

    Args:
        seed (int): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

#----------- Test functions -----------
if __name__ == "__main__":
    pass
