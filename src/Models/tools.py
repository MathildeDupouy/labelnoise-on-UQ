# by Mathilde Dupouy, 2024

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy import floor


def weights_init(m):
    """
    Xavier Normal initialization based on the layer type.
    (On the model of a work by Yamil Vindas)
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
        # nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)
        
def initialize_model_from_other(model, init_model_path, freeze=False):
    """Modify some of the model weights according to a given model.

    Args:
        model (torch.nn.Module): The model that will receive the weights
        init_model_path (str or Path): Path to the model that will transmit its weights
        freeze (bool, optional): Wether the loaded initialization model will be frozen or not. Defaults to False.

    Raises:
        NotImplementedError: If special cases of transmission are not implemented.
        FileNotFoundError: If the model path ids not found.

    Returns:
        torch.nn.Module: the modified model
    """
    init_model_path = Path(init_model_path)
    if not (init_model_path).exists():
        raise FileNotFoundError(f"Model for initialization not found at path {init_model_path}")
        
    try:
        prev_model = torch.load(init_model_path, weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load model from {init_model_path}: {e}")

    loaded_parameters = []
    if hasattr(model, 'embedding_net'):
        print("Model has embedding net")
        if hasattr(prev_model, 'embedding_net'):
            print("Prev model has embedding net")
            model.embedding_net = prev_model.embedding_net
        elif hasattr(prev_model, 'encoder'):
            print("Prev model has encoder")
            model.embedding_net = prev_model.encoder
        else:
            raise NotImplementedError(f"Model transmission not implemented from {prev_model.__class__.__name__} to {model.__class__.__name__}")
        loaded_parameters += list(model.embedding_net.parameters())
    elif hasattr(model, 'encoder'):
        print("Model has encoder")
        if hasattr(prev_model, 'embedding_net'):
            print("Prev model has embedding net")
            model.encoder = prev_model.embedding_net
        elif hasattr(prev_model, 'encoder'):
            print("Prev model has encoder")
            model.encoder = prev_model.encoder
            if hasattr(prev_model, 'decoder'):
                print("Prev model has decoder")
                model.decoder = prev_model.decoder
            loaded_parameters += list(model.decoder.parameters())
        else:
            raise NotImplementedError(f"Model transmission not implemented from {prev_model.__class__.__name__} to {model.__class__.__name__}")
        loaded_parameters += list(model.encoder.parameters())
    if freeze:
        for param in loaded_parameters:
            param.requires_grad = False
    return model
 
def toggle_dropout(model, activate=True):
    """
    Enable or disable dropout in the model, depending on the `activate` flag.
    This function loops through the model's layers and sets all Dropout layers
    to `train()` mode (for activation) or `eval()` mode (for deactivation).
    
    Args:
        model (nn.Module): The PyTorch model.
        activate (bool): If True, activate dropout by setting to train mode; 
                         if False, deactivate dropout by setting to eval mode.
    """
    # Iterate over all modules in the model (including at any depth of a submodule)
    for name, module in model.named_modules():
        # If the module is a Dropout layer, set it to train mode or eval mode
        if isinstance(module, nn.Dropout):
            if activate:
                module.train()  # Activate dropout (train mode)
            else:
                module.eval()   # Deactivate dropout (eval mode)