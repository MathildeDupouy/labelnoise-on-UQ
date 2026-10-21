import json
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE
from src.DataManipulation.representations import get_exp_representation

from src.Utils.vocabulary import ARGS, NAME, PROJECTION, REPRESENTATION


def get_projection(data, proj_config):
    """
    Project the data in a space using argument from a configuration dict.
    Arguments:
    ----------
        data: array [B, D1, D2, ...]
        Array (numpy or torch) representing the data to project, with B
        the number of elements and Di a latent space dimension.

        proj_config: dict
        A dictionary with the key "name" for the projection type and the key
        "args" with the arguments for the projection
    """
    print("======Computing projection=====")
    # Configure the TSNE object
    if proj_config[NAME].lower() == "tsne":
        projection_fn = TSNE(**proj_config[ARGS])
    # Project data
    data = data.reshape((data.shape[0], -1))
    projection = projection_fn.fit_transform(data)
    print("======Projection computed=====")
    return projection

def load_projection(projection_path):
    """
    Load a projection dictionary from a projection path.

    Arguments:
    ----------
        projection_path: str or Path
        Path to the pickle projection file
    Returns:
    --------
        dict: content of the pickle file
    """
    projection_path = Path(projection_path)
    assert projection_path.exists()

    with open(projection_path, 'rb') as f:
        projection_dict = pkl.load(f)
    return projection_dict

def get_exp_projection(
        exp_path, 
        model = None, 
        dataloaders = None, 
        device = None, 
        projection_name = None, 
        representation_name = "representations.pkl", 
        projections=None, 
        proj_config = None
    ):
    """
    Get (or compute) the representation from a trained experiment and project the data
    in a space and method described in the experiment configuration..
    
    Returned dictionary structure
    [mode_i] ('train', 'test', 'val')
        |_ 'dataset_index': array
        |_ 'labels'
            |_ [label_name_i]: array
        |_ 'projection': array<number of sample, reduced dim>
    'Projection params': 
        |_ 'name': string
        |_ 'args': dict<str, any>

    Arguments:
    ----------
        exp_path: str or Path
        Path to the experiment

        model: torch Model, default None
        If given, representations will be computed with this model

        dataloaders: dict of mode: torch.utils.DataLoader, default None
        If given, representations will be computed from these dataloaders

        device: torch device
        If given representations are loaded from/computed to this device

        projection_name: str, default None (projection.pkl hard coded)
        File name for the projection file (created if not existing)

        representation_name: str
        File name for the representation file loaded

        projections: array
        Projection array externally computed. If given, proj_config must be also given

        proj_config: dict
        A dictionary describing the projection parameters

    Returns:
    --------
        dict: The projection dictionary
        mode
            PROJECTION: Tensor with all the projections in a batch
            LABELS: dict of tensors of labels
            ID: list of ids
            SAMPLE: list of sample numbers
            "projection params": dictionary of the projection configuration
    """
    exp_path = Path(exp_path)
    config_path = exp_path/"config.json"

    assert exp_path.exists()
    assert config_path.exists()

    # Load configuration
    with config_path.open("r") as f:
        config_json = json.load(f)

    if projection_name is not None:
        projection_path = exp_path/projection_name
    else:
        projection_path = exp_path/"projection.pkl"

    if not projection_path.exists():
        representations_dict = get_exp_representation(exp_path, model, dataloaders, device, representation_name=representation_name)
        projections_dict = {}
        # Projection of all data then splitted by mode
        representations = []
        n_modes = [0]
        modes = list(representations_dict.keys())
        for mode in modes:
            # Initialize projection mode dict from representation dict
            projections_dict[mode] = {key: label for key, label in representations_dict[mode].items() if key != REPRESENTATION}

            # Get data
            representations.append(torch.Tensor(representations_dict[mode][REPRESENTATION]))
            n_modes.append(sum(n_modes) + np.array(representations_dict[mode][REPRESENTATION]).shape[0])
        # Project data
        if projections is None:
            representations = torch.cat(representations, 0)
            projections = get_projection(representations, config_json[PROJECTION])
        else:
            config_json[PROJECTION] = proj_config
        for i, mode in enumerate(modes):
            projections_dict[mode][PROJECTION] = projections[n_modes[i]:n_modes[i + 1], :]

        # Store projection params
        projections_dict["Projection params"] = config_json[PROJECTION]
        # Create projection file
        with open(projection_path, 'wb') as f:
            pkl.dump(projections_dict, f)
    else:
        # Reading the projection file already existing
        print(f"[projection] Projection path {projection_path.stem} for experiment {'/'.join(projection_path.parts[-3:-1])} already exists, loading it.")
        projections_dict = load_projection(projection_path)
    
    return projections_dict


###############################################################################
###############################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment projection help")
    parser.add_argument("-p",  "--exp_path", type=str, nargs="+", required=True,
                        help="Experiment paths to the experiment folders to be projected")

    args = parser.parse_args()
    exp_paths = args.exp_path
    for exp_path in exp_paths:
        projections_dict = get_exp_projection(exp_path)
        print(projections_dict["train"][PROJECTION].shape)

