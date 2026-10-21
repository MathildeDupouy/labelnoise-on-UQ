
"""
Run this script to launch an experiment from a configuration JSON file. 
See config_file_detail.md to know the different key and values necessary and available in a configuration file.
Arguments:
----------
    -e --experiments, int, multiple, required
    id of the experiments to run

    -n --number_run, int, optional (default 1)
    Number of training of each experiment
"""
import argparse
import json

# Add project root to sys.path
import os
import shutil
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in os.sys.path:
    os.sys.path.append(str(project_root))
from src.Experiments.experiments import (
    configJSON2obj,
    create_run_folder,
    get_dataloaders,
    get_model,
    get_optimizer,
)
from src.Experiments.train import train

from src.Models.tools import initialize_model_from_other
from src.Utils.tools import set_seed
from src.Utils.vocabulary import (
    ARGS,
    DATALOADER,
    DATASET,
    DEVICE,
    INFO,
    MODEL,
    OPTIMIZER,
    TEST,
    TRAIN,
    VAL,
)


def main_train_config(config_paths, exp_paths, number_run, init_model_path = None, init_optimizer_path = None, freeze_init=False, seeds=None, noise_seeds=None) -> None:
    """Load configuration paths and the associated experiments folders and train them number_run times.
    A model can be given to overwrite the configuration model.

    Args:
        config_paths (list<str or Path>): List of N configuration files paths
        exp_paths (list<str or Path>): List of N experiment folders paths
        number_run (int): Number of runs to independently train for each experiment
        init_model_path (torch.nn.Module, optional): A model to initialize the configuration model with. Defaults to None.
        init_optimizer_path (torch.optim.Optimizer, optional): An optimizer to initialize the configuration optimizer with. init_model_path must be provided.
        The optimizer should corresponds to all the model parameters. Defaults to None.
        freeze_init (boolean, optional): Wether the loaded initialization model will be frozen or not. Defaults to False.
        seeds (list<int>, optional): Seeds for the random number generators (initialization of model, torch, torch.cuda). If several seeds are given, they will be used for each run, depending on the run name. If None, no seed is given. Defaults to None.
        noise_seeds (list<int>, optional): Seeds for the noise generation. If several seeds are given, they will be used for each run of the experiments, depending on the run name. If None, no seed is given. Defaults to None.
    
    Returns:
        list<str or Path>: List of N*number_run paths to the trained experiments
    """
    run_paths = []
    for config_path, exp_path in zip(config_paths, exp_paths):
        config_path = Path(config_path)
        exp_path = Path(exp_path)
        assert config_path.exists(), f"Config path {config_path} does not exist."
        if not exp_path.exists():
            exp_path.mkdir(parents=True, exist_ok=True)
            print(f"[train_config] Experiment path {exp_path} created.")

        with Path(config_path).open("r") as f:
            json_config = json.load(f)

        # Translating to a configuration dict with instantiated objects
        obj_config = configJSON2obj(json_config)
        print(f"[Experiments] Asking for the device {json_config[DEVICE]}, final device {obj_config[DEVICE]}")
        
        # Transmitting a model weights if given
        if init_model_path is not None:
            model = initialize_model_from_other(obj_config[MODEL], init_model_path, freeze_init)
            obj_config[MODEL] = model

        # Transmitting an optimizer if given
        if init_optimizer_path is not None:
            assert init_model_path is not None, "init_model_path must be provided if init_optimizer_path is given."
            with Path(init_optimizer_path).open("rb") as f:
                optimizer = torch.load(f)
            obj_config[OPTIMIZER] = optimizer
        
        for i in range(number_run):
            print(f"=====Training experiment {exp_path.stem}, {i+1}th run======")
            run_path = create_run_folder(exp_path)
            run_path_nb = int(str(run_path.stem).split("-")[-1])
            
            # Save initialization model if given
            if init_model_path is not None:
                shutil.copy(Path(init_model_path), run_path)
                (Path(run_path)/'last_model.pt').rename(Path(run_path)/'init_model.pt')

            # Run seed if given
            if seeds is not None:
                if len(seeds) > 1:
                    assert len(seeds) > run_path_nb, f"[train_config] You did not give enough seeds, number of run is {run_path_nb} and number of seeds is {len(seeds)}."
                    seed = seeds[run_path_nb]
                else:
                    seed = seeds[0]
                print(f"[train_config] Setting seed {seed} for run {i+1}.")
                
                # Set seeds for reproducibility. Should be reset each run even if the same seed is given for each run, to avoid starting to another point in the random sequence for each run.
                set_seed(seed)
                
                # Store seed in the configuration for reproducibility
                json_config[INFO]["seed"] = seed
                
            # Data noise seed if given
            if noise_seeds is not None:
                if len(noise_seeds) > 1:
                    assert len(noise_seeds) >= run_path_nb, f"[train_config] You did not give enough noise seeds, number of run is {run_path_nb} and number of seeds is {len(noise_seeds)}."
                    noise_seed = noise_seeds[run_path_nb]
                else:
                    noise_seed = noise_seeds[0]
                print(f"[train_config] Setting noise seed {noise_seed} for run {i+1}.")
                
                # Store noise seed in the configuration for reproducibility
                json_config[DATASET][ARGS]["seed"] = noise_seed
                
                # Regenerating dataset with the new noise seed
                obj_config[DATASET] = {}
                obj_config[DATALOADER], obj_config[DATASET][TRAIN], obj_config[DATASET][VAL], obj_config[DATASET][TEST] = get_dataloaders(json_config)
            
            # Reset model and optimizer for each run
            obj_config[MODEL] = get_model(json_config)
            obj_config[OPTIMIZER] = get_optimizer(json_config, obj_config[MODEL])
            
            

            # Save configuration
            with (Path(run_path)/"config.json").open("w") as f:
                json.dump(json_config, f, indent = 4)

            # Training
            train(obj_config, run_path)
            
            run_paths.append(run_path)
    return run_paths
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments test help")
    parser.add_argument("-p",  "--config_path", type=str, nargs="+", required=True,
                        help="Path to the JSON configuration file")
    parser.add_argument("-e",  "--exp_path", type=str, nargs="+", required=True,
                        help="Path to the experiment folder, where the experiments will be saved.")
    parser.add_argument("-n",  "--number_run", type=int, required=False, default=1,
                        help="Number of runs for each experiments")
    parser.add_argument("-m",  "--init_model_path", type=str, required=False, default=None,
                        help="Path to a .pt ot .pth model to be used for training initialization")
    parser.add_argument("-o",  "--init_optimizer_path", type=str, required=False, default=None,
                        help="Path to a .pt ot .pth optimizer to be used for training initialization")
    parser.add_argument("-f",  "--freeze_init", type=bool, required=False, default=False,
                        help="Wether the loaded initialization model will be frozen or not")
    parser.add_argument("-s", "--seed", type=int, nargs="+", required=False, default=None,
                        help="Seed for the random number generators. If several seeds are given, they will be used for each run of the experiments in order. If None, random seeds will be used.")
    parser.add_argument("--test", action="store_true", help="Whether to run the script in test mode, to go through unit tests.")

    args = parser.parse_args()

    # Classical train_config execution
    if not args.test:
        assert len(args.exp_path) == len(args.config_path), f"You did not give the same number of configuration and experiment paths ({len(args.config_path)} configs, {len(args.exp_path)} paths)."

        main_train_config(config_paths=args.config_path,
                        exp_paths=args.exp_path,
                        number_run=args.number_run,
                        init_model_path=args.init_model_path,
                        init_optimizer_path=args.init_optimizer_path,
                        freeze_init=args.freeze_init,
                        seeds=args.seed)
    
    else:
        #------------------------- Unit tests ---------------------------
        # Test reproducibility of initialization with a seed
        print("--------- Testing reproducibility of initialization with a seed... --------------")
        config_path = Path(args.config_path[0])
        exp_path = Path(args.exp_path[0])
        assert config_path.exists(), f"Config path {config_path} does not exist."
        if not exp_path.exists():
            exp_path.mkdir(parents=True)
            print(f"[train_config] Experiment path {exp_path} created.")

        with Path(config_path).open("r") as f:
            json_config = json.load(f)

        # Translating to a configuration dict with instantiated objects
        obj_config = configJSON2obj(json_config)
        print(f"[Experiments] Asking for the device {json_config[DEVICE]}, final device {obj_config[DEVICE]}")

        number_run = args.number_run
        for i in range(number_run):
            print(f"=====Training experiment {exp_path.stem}, {i+1}th run======")
            run_path = create_run_folder(exp_path)

            # Initialize models
            seed = None
            seeds = args.seed
            if seeds is not None:
                if len(seeds) > 1:
                    assert len(seeds) >= number_run, f"[train_config] You did not give enough seeds for the number of runs ({len(seeds)} seeds, {number_run} runs)."
                    seed = seeds[i]
                else:
                    seed = seeds[0]
                print(f"[train_config] Setting seed {seed} for run {i+1}.")
                
                # Set seeds for reproducibility. Should be reset each run even if the same seed is given for each run, to avoid starting to another point in the random sequence for each run.
                set_seed(seed)
                
                # Store seed in the configuration for reproducibility
                json_config[INFO]["seed"] = seed

            obj_config[MODEL] = get_model(json_config)
            obj_config[OPTIMIZER] = get_optimizer(json_config, obj_config[MODEL])

            # Save configuration
            with (Path(run_path)/f"config_{i}_seed{seed}.json").open("w") as f:
                json.dump(json_config, f, indent = 4)
            
            # Save models
            # with (Path(run_path)/f"model_{i}_seed{seed}.pt").open("w") as f:
            torch.save(obj_config[MODEL], Path(run_path)/f"model_{i}_seed{seed}.pt")
            
            # Print first and last weights of the model to compare
            n_w = 1
            w = 0
            print(f">>>>> Iteration {i} - seed {seed}")
            for param in obj_config[MODEL].parameters():
                print(param.shape, param.data)
                w += 1
                if w >= n_w:
                    break
