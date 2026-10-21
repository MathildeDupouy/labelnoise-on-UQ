"""Script to evaluate a model with a specific uq configuration. 
It also applies the UQ configuration necessary modifications to the model and so on if needed.
The UQ config has to contain the UQ config part with the UQ keyword, and can contain other configuration modifications.
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
from src.Experiments.evaluate import evaluate
from src.Experiments.experiments import configJSON2obj
from src.Experiments.inference import inference

from src.Utils.vocabulary import ARGS, EVALUATION, MODEL, NAME, NB_INFERENCE, UQ

###############################################################################
###############################################################################

def main_evaluate_uq(parent_folder, uq_config_path, all, exp_folders = None, metric_config = None):
    # Getting folders
    parent_folder = Path(parent_folder)
    assert parent_folder.exists(), f"Folder {parent_folder} does not exist."
    if all :
        folders = sorted([parent_folder/exp_folder for exp_folder in parent_folder.glob('*') if (parent_folder/exp_folder).is_dir()])
    else:
        assert exp_folders is not None, "Please enter run(s) folder(s) if '--all' is set to false"
        exp_folders = exp_folders
        folders = [parent_folder/exp_folder for exp_folder in exp_folders]

    # Getting external metric config if provided
    uq_config_path = Path(uq_config_path)
    assert uq_config_path.exists(), f"[evaluate_uq] UQ config {uq_config_path} does not exist."
    print(f"[evaluate_uq] Using UQ list from {uq_config_path.parent.stem}/{uq_config_path.stem}")
    with uq_config_path.open("r") as f:
        uq_config = json.load(f)
    # Getting uq description
    n_inference = uq_config[NB_INFERENCE]
    uq_names = []
    uq_args = []
    n_uq_methods = len(uq_config[UQ])
    for i_uq in range(n_uq_methods):
        uq_names.append(uq_config[UQ][i_uq][NAME])
        if ARGS in uq_config[UQ][i_uq].keys():
            uq_args.append(uq_config[UQ][i_uq][ARGS])
        else:
            uq_args.append({})

    # Getting external metric config if provided
    if metric_config is not None:
        metric_path = Path(metric_config)
        assert metric_path.exists(), f"[evaluate_uq_some_metrics] Metric config {metric_path} does not exist."
        print(f"[evaluate] Using metric list from {metric_path.parent.stem}/{metric_path.stem}")
        with metric_path.open("r") as f:
            metric_config = json.load(f)
    else:
        metric_config = None

    #-------- Evaluation
    for run_path in folders:
        print(f"=====Evaluating UQ experiment {run_path.parent.stem}-{run_path.stem}======")
        
        for i_uq in range(n_uq_methods):

            #--- Getting config
            with open(run_path/"config.json", "r") as f:
                json_config = json.load(f)
            # Replacing uq config (and other params if detailed such as n_inference)
            for key, val in uq_config.items():
                json_config[key] = val
            #--- Modifying model if needed
            # MC dropout test dropout
            print(uq_names[i_uq].lower(), uq_args[i_uq])
            if uq_names[i_uq].lower() == "mcdropout entropy" and "test dropout" in uq_args[i_uq].keys():
                print("1. Modifying test dropout")
                print("2", json_config[MODEL])
                for key, val in json_config[MODEL][ARGS].items():
                    if "dropout" in key.lower(): # for each dropout accessible parameter
                        json_config[MODEL][ARGS][key] = uq_args[i_uq]["test dropout"]
                        print("3", json_config[MODEL])
            #--- Instantiating config
            obj_config = configJSON2obj(json_config)
            # Loading trained parameters
            obj_config[MODEL].load_state_dict(torch.load(run_path/"last_model.pt", weights_only=True))
            
            # Replacing metric
            if metric_config is not None:
                obj_config[EVALUATION] = metric_config[EVALUATION] if EVALUATION in metric_config.keys() else metric_config

            print("|||||||||||||||||||", obj_config["device"])
            print(f"[evaluate_uq] Train dataset size: {len(obj_config['dataset']['train'])}")
            print(f"[evaluate_uq] Test dataset size: {len(obj_config['dataset']['test'])}")

            # Load predictions
            repr_path = run_path/f"representations_{uq_names[i_uq].replace(' ', '-')}_ni{n_inference}.pkl"
            pred_path = run_path/f"predictions_{uq_names[i_uq].replace(' ', '-')}_ni{n_inference}.pkl"
            #--- Loading predictions
            if pred_path.exists():
                with pred_path.open("rb") as f:
                    predictions = pkl.load(f)
            
            #--- Inference to get predictions
            else:
                if hasattr(obj_config[MODEL], 'latent_representation'):
                    predictions, representations = inference(obj_config)
                    # Saving representations
                    if not repr_path.exists():
                        print(f"[evaluate_uq] Saving {repr_path}")
                        with (repr_path).open("wb") as f:
                            pkl.dump(representations, f)
                else:
                    predictions = inference(obj_config)
                # Saving predictions
                with (pred_path).open("wb") as f:
                    print(f"[evaluate_uq] Saving {pred_path}")
                    pkl.dump(predictions, f)

            # Load representations
            representations = None
            # Representations path may not exist for model without latent space
            metric_path = run_path/f"metric_{uq_names[i_uq].replace(' ', '-')}_ni{n_inference}.json"
            if repr_path.exists():
                with repr_path.open("rb") as f:
                    representations = pkl.load(f) # Overwrite representations if computed in the previous "else"
                metrics = evaluate(obj_config, run_path, predictions, representations, metric_path_name=metric_path)
            else: 
                metrics = evaluate(obj_config, run_path, predictions, metric_path_name=metric_path)

    return metrics

if __name__ == "__main__":
    """
    Script to evaluate an experiment given its configuration and an uq configuration
    Usage: python src/Scripts/evaluate_uq.py -p /path/to/experiments/Exp_ClassifESRmulti_2DCNN_balancedCE_dropout/noise_rate_0-0/ -a -uq /path/to/configs/ESR_2DCNN/mcdropout_config_multi-target.json
    """
    parser = argparse.ArgumentParser(description="Evaluation script help")
    parser.add_argument("-p",  "--parent_folder", type=str, required=True,
                        help="Path to the experiment folder")
    parser.add_argument("-uq",  "--uq_config", type=str, required=True, default=None,
                        help="Path to the uncertainty quantification configuration (json)")
    parser.add_argument("-a",  "--all",  action="store_true", required=False, default=False,
                        help="If all the runs in the parent folder have to be considered.")
    parser.add_argument("-f",  "--exp_folders", type=str, nargs="+", required=False,
                        help="Path to the experiments folder from the root folder, if all is false.")
    parser.add_argument("-m",  "--metric_config", type=str, required=False, default=None,
                        help="Path to the evaluation configuration (json)")

    # Interpreting arguments
    args = parser.parse_args()

    main_evaluate_uq(args.parent_folder, args.uq_config, args.all, args.exp_folders, args.metric_config)

