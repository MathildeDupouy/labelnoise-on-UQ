"""
Run this script to infer a model from a dataset.
Arguments:
----------
    -p --parent_folder, str, required
    path to the parent folder of the run

    -o --output_path, str, required
    path to the output folder for predictions
    
    -m --model, str, required
    path ot the model path

    -dc --dataset_config, str, optional (default None)
    Path to the JSON file with the configuration from the dataset
    Exemple :
    {
        "dataset type": "HITS_2D",
        "data path": "/path/to/data/dataset_name/data.hdf5",
        "initial_duration": 1500,
        "duration": 400,
        "multiple": 16,
        "args": {
            "remove_pos_U": false,
            "remove_class_U": false
        }
    }

    -uc --uncertainty_config, str, optional (default None)
    Path to the JSON file with the configuration of uncertainty evaluation
    
    -mc --model_config, str, optional (default None)
    Path to the JSON file with the configuration of the model for evaluation.
    This can be useful for example if the dropout need to be set for inference.
"""

import argparse
import json

# Add project root to sys.path
import os
import pickle as pkl
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in os.sys.path:
    os.sys.path.append(str(project_root))
from src.Experiments.experiments import configJSON2obj
from src.Experiments.inference import inference

from src.Utils.vocabulary import DATASET, DEVICE, MODEL, NB_INFERENCE, UQ

if __name__ == "__main__":
    # Example : python src/Scripts/infer.py [-dc /path/to/configs/dataset/dataset_config.json]
    #                                       [-uc /path/to/configs/uncertainty/uncertainty_config.json]
    #                                       -p /path/to/experiments/experiment_name/run-0
    #                                       -m /path/to/experiments/experiment_name/run-0/last_model.pt
    #                                       -o /path/to/experiments/inferences/temporal_sub_400ms_a1_b0-01_run-0
    
    # ----------- Define arguments
    parser = argparse.ArgumentParser(description="Inference script help")
    parser.add_argument("-p",  "--parent_folder", type=str, required=True,
                        help="Path to the experiment folder")
    parser.add_argument("-o",  "--output_path", type=str, required=True,
                        help="Path to the folder where predictions will be stored")
    parser.add_argument("-m",  "--model", type=str, required=True,
                        help="Path to the model which STATE ONLY will be used")
    parser.add_argument("-dc",  "--dataset_config", type=str, required=False, default=None,
                        help="Path to the JSON file with the configuration from the dataset")
    parser.add_argument("-uc",  "--uncertainty_config", type=str, required=False, default=None,
                        help="Path to the JSON file with the configuration of uncertainty evaluation")
    parser.add_argument("-mc",  "--model_config", type=str, required=False, default=None,
                        help="Path to the JSON file with the configuration of the model for evaluation")

    args = parser.parse_args()
    exp_folder = Path(args.parent_folder)
    model_path = Path(args.model)
    assert exp_folder.exists(), f"{exp_folder} does not exist."
    assert model_path.exists(), f"{model_path} does not exist."
    output_path = Path(args.output_path)
    os.mkdir(output_path)

    # ----------- Load config and replace given arguments
    with (exp_folder/'config.json').open("r") as f:
        json_config = json.load(f)

    # Replacing dataset arguments
    if args.dataset_config is not None :
        with open(args.dataset_config, "r") as f:
            dataset_config = json.load(f)
        if DATASET in dataset_config.keys():
            json_config[DATASET] = dataset_config[DATASET]
        else:
            json_config[DATASET] = dataset_config

    # Replacing uncertainty arguments
    if args.uncertainty_config is not None :
        with open(args.uncertainty_config, "r") as f:
            uncertainty_config = json.load(f)
        json_config[UQ] = uncertainty_config[UQ]
        json_config[NB_INFERENCE] = uncertainty_config[NB_INFERENCE]

    # Replacing model arguments
    if args.model_config is not None :
        with open(args.model_config, "r") as f:
            model_config = json.load(f)
        json_config[MODEL] = model_config[MODEL]

    # ----------- Instantiating configuration
    # Translating to a configuration dict with instantiated objects
    obj_config = configJSON2obj(json_config)
    print(f"[Experiments] Asking for the device {json_config[DEVICE]}, final device {obj_config[DEVICE]}")

    ## Replacing model state by the loaded one
    model_state = torch.load(model_path, weights_only=False).state_dict()
    obj_config[MODEL].load_state_dict(model_state)
    model = obj_config[MODEL]
    
    # ----------- Inference
    if hasattr(model, 'latent_representation'):
        # Inference
        predictions, representations = inference(obj_config)
        with (output_path/"representations.pkl").open("wb") as f:
            pkl.dump(representations, f)
    else:
        predictions = inference(obj_config)

    # ----------- Store results
    with (output_path/"predictions.pkl").open("wb") as f:
        pkl.dump(predictions, f)
    with (output_path/"config.json").open("w") as f:
        json.dump(json_config, f, indent = 4)
