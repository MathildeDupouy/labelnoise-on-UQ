import argparse
import json
from pathlib import Path

from src.Scripts.evaluate_exp import main_evaluate_exp
from src.Scripts.evaluate_uq import main_evaluate_uq
from src.Scripts.train_config import main_train_config
from src.Utils.synthetize_exp_evol import ExperimentEvolution
from src.Utils.vocabulary import (
    ARGS,
    CHECKPOINT_FREQ,
    DATASET,
    NB_EPOCHS,
    TEST,
    TRAIN,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments test help")
    parser.add_argument("-c", "--config_path", type=str, required=True,
                        help="Path to the configuration folder for ESR")
    parser.add_argument("-e", "--exp_path", type=str, required=True,
                        help="Path to the experiment folder for ESR with memorization")
    parser.add_argument("-n",  "--n_runs", type=int, required=True,
                        help="Number of runs")
    parser.add_argument("-pts",  "--n_points", type=int, required=False, default=50,
                        help="Number of points by training")
    args = parser.parse_args()
    
    #Define experiments to run
    config_path = Path(args.config_path)
    assert config_path.exists(), f"Config path {config_path} does not exist"
    exp_path = Path(args.exp_path)
    assert exp_path.exists(), f"Exp path {exp_path} does not exist"
    exp_list = [
        (
            config_path/"Exp_ClassifESRmulti_1DCNNTransf_balancedCE_nodropout.json",
            exp_path/"Exp_ClassifESRmulti_1DCNNTransf_balancedCE_nodropout"
        ),
        (
            config_path/"Exp_ClassifESRmulti_1DCNNTransf_balancedCE_dropout.json",
            exp_path/"Exp_ClassifESRmulti_1DCNNTransf_balancedCE_dropout"
        ),
        (
            config_path/"Exp_ClassifESRmulti_1DCNNTransf_balancedGCE_q0-4_nodropout.json",
            exp_path/"Exp_ClassifESRmulti_1DCNNTransf_balancedGCE_q0-4_nodropout"
        ),
        (
            config_path/"Exp_ClassifESRmulti_1DCNNTransf_balancedGCE_q0-4_dropout.json",
            exp_path/"Exp_ClassifESRmulti_1DCNNTransf_balancedGCE_q0-4_dropout"
        ),
        (
            config_path/"Exp_ClassifESRmulti_1DCNNTransf_balancedGCE_q0-7_nodropout.json",
            exp_path/"Exp_ClassifESRmulti_1DCNNTransf_balancedGCE_q0-7_nodropout"
        ),
        (
            config_path/"Exp_ClassifESRmulti_1DCNNTransf_balancedGCE_q1_nodropout.json",
            exp_path/"Exp_ClassifESRmulti_1DCNNTransf_balancedGCE_q1_nodropout"
        ),
    ]
    uq_path = config_path/"mcdropout_config_multi-target.json"
    nr_list = [0.0, 0.2, 0.4, 0.6]
    n_runs = args.n_runs
    n_pts = args.n_points
    seeds_runs = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    seeds_noise = [141, 142, 143, 144, 145, 146, 147, 148, 149, 150]
    
    # run experiments
    config_list = [exp_config for exp_config, _ in exp_list]
    prev_configs = []
    for nr in nr_list:
        # Go through configs to do some modifications
        for config_path in config_list:
            with config_path.open("r") as f:
                config_dict = json.load(f)
            prev_configs.append(config_dict)
            # Modify noise rate
            config_dict[DATASET][ARGS]["noise_rate"] = nr
            # Modify checkpoint frequency to have ~n_pts points
            config_dict[CHECKPOINT_FREQ] = max(1, int(round(int(config_dict[NB_EPOCHS]) / n_pts)))
            print(f"[script] Checkpoint freq. will be {config_dict[CHECKPOINT_FREQ]}.")
    
        # Exp path refinement
        path_list = [exp_dir/f"noise_rate_{str(nr).replace(".", "-")}" for _, exp_dir in exp_list]
        
        # Training
        run_paths = main_train_config(
            config_paths = config_list,
            exp_paths = path_list,
            number_run = n_runs,
            seeds = seeds_runs,
            noise_seeds = seeds_noise
        )
        
        for parent_exp in path_list:
            main_evaluate_exp(parent_exp, True)
            main_evaluate_uq(parent_exp, uq_path, True)
        
        # Store a synthetic summary of the evolution of the scores and accuracies for each experiment
        for run_path in run_paths:
            for mode in [TRAIN, TEST]:
                evolution = ExperimentEvolution(run_path, mode=mode)
                evolution.get_score_evolution()
                evolution.get_acc_evolution()
                evolution.store_results()
        
        for config_path, prev_config in zip(config_list, prev_configs):
            with config_path.open("w") as f:
                json.dump(prev_config, f, indent = 4)