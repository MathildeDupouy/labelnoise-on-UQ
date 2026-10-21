import argparse
import json

# Add project root to sys.path
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in os.sys.path:
    os.sys.path.append(str(project_root))
from src.Scripts.evaluate_exp import main_evaluate_exp
from src.Scripts.evaluate_uq import main_evaluate_uq
from src.Scripts.train_config import main_train_config
from src.Utils.vocabulary import ARGS, DATASET

if __name__ == "__main__":
    """Example usage: python configs/MNIST/exp_MNIST.py -c configs/MNIST/ -e experiments/MNIST -n 5
    """
    parser = argparse.ArgumentParser(description="Experiments test help")
    parser.add_argument("-c", "--config_path", type=str, required=True,
                        help="Path to the configuration folder for MNIST")
    parser.add_argument("-e", "--exp_path", type=str, required=True,
                        help="Path to the experiment folder for MNIST")
    parser.add_argument("-n",  "--n_runs", type=int, required=True,
                        help="Number of runs")
    args = parser.parse_args()
    
    #Define experiments to run
    config_path = Path(args.config_path)
    assert config_path.exists(), f"Config path {config_path} does not exist"
    exp_path = Path(args.exp_path)
    assert exp_path.exists(), f"Exp path {exp_path} does not exist"
    
    #Define experiments to run
    config_path = Path(args.config_path)
    assert config_path.exists(), f"Config path {config_path} does not exist"
    exp_path = Path(args.exp_path)
    assert exp_path.exists(), f"Exp path {exp_path} does not exist"
    exp_list = [
        (
            config_path/"Exp_ClassifMNIST_2DCNN_CE_nodropout.json",
            exp_path/"Exp_ClassifMNIST_2DCNN_CE_nodropout"
        ),
        (
            config_path/"Exp_ClassifMNIST_2DCNN_CE_dropout.json",
            exp_path/"Exp_ClassifMNIST_2DCNN_CE_dropout"
        ),
        (
            config_path/"Exp_ClassifMNIST_2DCNN_GCE_q0-4_nodropout.json",
            exp_path/"Exp_ClassifMNIST_2DCNN_GCE_q0-4_nodropout"
        ),
        (
            config_path/"Exp_ClassifMNIST_2DCNN_GCE_q0-4_dropout.json",
            exp_path/"Exp_ClassifMNIST_2DCNN_GCE_q0-4_dropout"
        ),
        (
            config_path/"Exp_ClassifMNIST_2DCNN_GCE_q0-7_nodropout.json",
            exp_path/"Exp_ClassifMNIST_2DCNN_GCE_q0-7_nodropout"
        ),
        (
            config_path/"Exp_ClassifMNIST_2DCNN_GCE_q0-7_dropout.json",
            exp_path/"Exp_ClassifMNIST_2DCNN_GCE_q0-7_dropout"
        ),
        (
            config_path/"Exp_ClassifMNIST_2DCNN_GCE_q1_nodropout.json",
            exp_path/"Exp_ClassifMNIST_2DCNN_GCE_q1_nodropout"
        ),
    ]
    uq_path = config_path/"mcdropout_config_hard-target.json"
    nr_list = [0.0, 0.2, 0.4, 0.6]
    n_runs = args.n_runs
    seeds_runs = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    seeds_noise = [140, 141, 142, 143, 144, 145, 146, 147, 148, 149]
    
    # run experiments
    config_list = [exp_config for exp_config, _ in exp_list]
    for nr in nr_list:
        # Go through configs to modify the noise rate
        prev_configs = []
        for config_path in config_list:
            with config_path.open("r") as f:
                config_dict = json.load(f)
            prev_configs.append(config_dict)
            config_dict[DATASET][ARGS]["noise_rate"] = nr
            with config_path.open("w") as f:
                json.dump(config_dict, f, indent = 4)
    
        # Exp path refinement
        path_list = [exp_dir/f"noise_rate_{str(nr).replace(".", "-")}" for _, exp_dir in exp_list]
        
        # Training
        main_train_config(
            config_paths = config_list,
            exp_paths = path_list,
            number_run = n_runs,
            seeds = seeds_runs,
            noise_seeds = seeds_noise
        )
        
        for parent_exp in path_list:
            main_evaluate_exp(parent_exp, True)
            main_evaluate_uq(parent_exp, uq_path, True)
        
        for config_path, prev_config in zip(config_list, prev_configs):
            with config_path.open("w") as f:
                json.dump(prev_config, f, indent = 4)