# TRANSLATE THE CONFIG FILE
import datetime
import os
from pathlib import Path

import git
import numpy as np
import torch
from labml_nn.optimizers import noam
from sklearn.utils.class_weight import compute_class_weight
from src.DataManipulation.balance_dataset import balance_by_subsampling
from src.DataManipulation.ECG_Dataset import ECG_Dataset, ptb_split_train_test
from src.DataManipulation.EEG_Dataset import EEG_Dataset, esr_split_train_test
from src.DataManipulation.MNIST_Dataset import MNISTDataset
from torch.utils.data import DataLoader

from src.Models.CNNs.time_frequency_simple_CNN import TimeFrequency2DCNN
from src.Models.tools import weights_init
from src.Models.Transformers.Transformer_Encoder_RawAudioMultiChannelCNN import (
    TransformerClassifierMultichannelCNN,
)
from src.Utils.GCE import GeneralizedCrossEntropyLoss
from src.Utils.GeoJSD import GeoJSDLoss
from src.Utils.uncertainty import (
    aleatoric_uncertainty_entropy,
    epistemic_uncertainty_entropy,
    total_uncertainty_entropy,
)
from src.Utils.vocabulary import *
from src.Utils.weightedMAE import WeightedMAELoss


def get_dataloaders(json_config):
    """
    Get information from the config DATASET, TRAIN_LABELS, TEST_LABELS and EVALUATION keys to build 
    the corresponding dataloaders.

    Arguments:
    ----------
        json_config: dict
        configuration file
    
    Returns:
    --------
        dict: keys are TRAIN and TEST and their values are torch dataloaders
    """
    # Create datasets and dataloaders
    dataloaders = {}
    train_dataset, test_dataset, val_dataset = None, None, None
    train_labels = list(set(json_config[TRAIN_LABELS][LABELS] + json_config[EVALUATION][ALL][LABELS] + json_config[EVALUATION][TRAIN][LABELS]))
    if VAL in json_config[EVALUATION].keys():
        val_labels = list(set(json_config[TRAIN_LABELS][LABELS] + json_config[EVALUATION][ALL][LABELS] + json_config[EVALUATION][VAL][LABELS]))
    else:
        val_labels = list(set(json_config[TRAIN_LABELS][LABELS] +json_config[EVALUATION][ALL][LABELS]))
    test_labels = list(set(json_config[TEST_LABELS] + json_config[EVALUATION][ALL][LABELS] + json_config[EVALUATION][TEST][LABELS]))
    
    # ----- PTB dataset
    if "ptb" in json_config[DATASET][DATASET_TYPE].lower():
        train_list, test_list = ptb_split_train_test(json_config[DATASET][ABNORMAL_PATH], json_config[DATASET][NORMAL_PATH], test_size=json_config[DATASET][TEST_SIZE])
        # Sub sample train set to balance dataset
        if "sub" in json_config[DATASET][DATASET_TYPE].lower():
            train_list = balance_by_subsampling(train_list, seed=json_config[DATASET]["seed"])
        # Create dataset
        train_dataset = ECG_Dataset(
            train_list,
            label_keys=train_labels,
            **json_config[DATASET][ARGS]
            )
        # No noise in the test dataset + normalization as in the train dataset
        train_mean, train_std = train_dataset.mean, train_dataset.std
        json_config[DATASET][ARGS]['normalize_mean_std'] = [float(train_mean), float(train_std)]
        test_dataset = ECG_Dataset(
            test_list,
            label_keys=test_labels,
            **{k: v for k, v in json_config[DATASET][ARGS].items() if k not in ['noise_rate']}
            )
        json_config[DATASET][INFO] = {
            "train size": len(train_dataset),
            "train sample size": train_dataset.signal_length,
            "test size": len(test_dataset),
            "test sample size": test_dataset.signal_length,
            "normalize_mean_std": [float(train_mean), float(train_std)],
            "noise seed (train)": train_dataset.seed
        }
    
    # ----- ESR dataset
    elif "esr" in json_config[DATASET][DATASET_TYPE].lower():
        train_list, test_list = esr_split_train_test(json_config[DATASET][CSV_PATH], test_size=json_config[DATASET][TEST_SIZE])
        if VAL_SIZE in json_config[DATASET].keys():
            val_size = json_config[DATASET][VAL_SIZE]
            n_val = int(len(train_list) * val_size)
            val_list = train_list[:n_val]
            train_list = train_list[n_val:]
        # Create dataset
        train_dataset = EEG_Dataset(
            train_list,
            label_keys=train_labels,
            **json_config[DATASET][ARGS]
            )
        train_mean, train_std = train_dataset.mean, train_dataset.std
        # Noise in the validation set + normalization as in the train dataset
        if VAL_SIZE in json_config[DATASET].keys():
            json_config[DATASET][ARGS]['normalize_mean_std'] = [float(train_mean), float(train_std)]
            val_dataset = EEG_Dataset(
                val_list,
                label_keys=val_labels,
                **json_config[DATASET][ARGS]
                )
        # No noise in the test dataset + normalization as in the train dataset
        json_config[DATASET][ARGS]['normalize_mean_std'] = [float(train_mean), float(train_std)]
        test_dataset = EEG_Dataset(
            test_list,
            label_keys=test_labels,
            **{k: v for k, v in json_config[DATASET][ARGS].items() if k not in ['noise_rate']}
            )
        json_config[DATASET][INFO] = {
            "train size": len(train_dataset),
            "train sample size": train_dataset.signal_length,
            "validation size": len(val_dataset) if val_dataset is not None else 0,
            "validation sample size": val_dataset.signal_length if val_dataset is not None else 0,
            "test size": len(test_dataset),
            "test sample size": test_dataset.signal_length,
            "normalize_mean_std": [float(train_mean), float(train_std)],
            "noise seed (train)": train_dataset.seed
        }

    # ----- MNIST dataset
    elif "mnist" in json_config[DATASET][DATASET_TYPE].lower():
        val_size = 0.
        if VAL_SIZE in json_config[DATASET].keys():
            val_size = json_config[DATASET][VAL_SIZE]
            # Noise in the validation set (no normalization implemented for MNIST, only default (0.5, 0.5) normalization)
            val_dataset = MNISTDataset(json_config[DATASET][DATA_PATH], label_keys=val_labels, train=True, val = True, val_ratio=val_size, **json_config[DATASET][ARGS])
        train_dataset = MNISTDataset(json_config[DATASET][DATA_PATH], label_keys=train_labels, train=True, val = False, val_ratio=val_size, **json_config[DATASET][ARGS])
        test_dataset = MNISTDataset(
            json_config[DATASET][DATA_PATH], 
            label_keys=test_labels, train=False, 
            **{k: v for k, v in json_config[DATASET][ARGS].items() if k not in ['noise_rate']}
            )
        json_config[DATASET][INFO] = {
            "train size": len(train_dataset),
            "validation size": len(val_dataset) if val_dataset is not None else 0,
            "test size": len(test_dataset),
            "noise seed (train)": train_dataset.seed
            }
    else:
        raise NotImplementedError(f"[Experiments] Dataset config {json_config[DATASET][DATASET_TYPE]} is not implemented.")

    # Dataloader prepared for training and evaluation
    # Get batch size
    if type(json_config[BATCH_SIZE]) == int:
        train_batch_size = json_config[BATCH_SIZE]
        val_batch_size = json_config[BATCH_SIZE]
        test_batch_size = json_config[BATCH_SIZE]
    elif type(json_config[BATCH_SIZE]) == list and len(json_config[BATCH_SIZE]) == 2:
        train_batch_size = json_config[BATCH_SIZE][0]
        val_batch_size = json_config[BATCH_SIZE][0]
        test_batch_size = json_config[BATCH_SIZE][1]
    elif type(json_config[BATCH_SIZE]) == list and len(json_config[BATCH_SIZE]) == 3:
        train_batch_size = json_config[BATCH_SIZE][0]
        val_batch_size = json_config[BATCH_SIZE][1]
        test_batch_size = json_config[BATCH_SIZE][2]
    else:
        raise ValueError(f"[Experiments] batch size config {json_config[BATCH_SIZE]} is not valid. It should be either an int, or a list of 2 or 3 ints for respectively train and test batch size, or train, val and test batch size.")
    # Instatiate dataloaders
    n_train = len(train_dataset)
    if n_train > 0: #RandomSampler do not support empty datasets
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=torch.utils.data.RandomSampler(train_dataset, replacement=False))
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size)
    n_val = len(val_dataset) if val_dataset is not None else 0
    if n_val > 0:
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, sampler=torch.utils.data.RandomSampler(val_dataset, replacement=False))
        dataloaders[VAL] = val_loader
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
    dataloaders[TRAIN] = train_loader
    dataloaders[TEST] = test_loader

    return dataloaders, train_dataset, val_dataset, test_dataset

def get_model(json_config):
    """
    Get a model from the configuration dict, and initialize it.
    """
    if json_config[MODEL][NAME].lower() == "timefreq2dcnn":
        model = TimeFrequency2DCNN(**json_config[MODEL][ARGS])
    elif json_config[MODEL][NAME].lower() == "time1dcnntransformer":
        # From Yamil's code, see article "[1] Y. Vindas, B. K. Guépié, M. Almar, E. Roux, and P. Delachartre, “An hybrid cnn-transformer model based on multi-feature extraction and attention fusion mechanism for cerebral emboli classification,” in Machine Learning for Healthcare, Durham, United States, Aug. 2022. Available: https://hal.science/hal-04429525"
        model = TransformerClassifierMultichannelCNN(**json_config[MODEL][ARGS])
    else:
        raise NotImplementedError(f"[Experiments] Model config {json_config[MODEL][NAME]} is not implemented.")
    
    # Initialize weights
    model.apply(weights_init)
    return model

def get_optimizer(json_config, model):
    """
    Instantiate an optimizer from the configuration dict and a given model.
    
    Args:
        json_config (dict): configuration dict
        model (torch.nn.Module): model to optimize, i.e. from which the parameters are get to instantiate the optimizer.
    
    Returns:
        torch.optim.Optimizer: the instantiated optimizer
    """
    if json_config[OPTIMIZER][NAME].lower() == "adam":
        # Adaptative Moment Estimation (Kingma and Ba, 2014) is an optimization algorithm that computes adaptive learning rates for each parameter, using a bias-corrected moment estimation.
        optimizer = torch.optim.Adam(model.parameters(), **json_config[OPTIMIZER][ARGS])
    elif json_config[OPTIMIZER][NAME].lower() == "adamax":
        # Adamax uses the infinite norm (maximum norm) instead of the L2 norm for computing the second moment, which provides a more robust measure of the gradients’ magnitude. 
        # This helps to mitigate issues like vanishing and exploding gradients and enables a better handling of extreme gradients and stabilization of the the training process (especially for complex architectures).
        optimizer = torch.optim.Adamax(model.parameters(), **json_config[OPTIMIZER][ARGS])
    elif json_config[OPTIMIZER][NAME].lower() == "noam":
        # Noam is a custom lr schedule originally introduced in Vaswani et al. 2017 that wraps around Adam.
        # It can be reproduced in Pytorch, here we use the labml_nn implementation for simplicity.
        optimizer = noam.Noam(params = model.parameters(), d_model=json_config[MODEL][ARGS]["d_model"], **json_config[OPTIMIZER][ARGS])
    else:
        raise NotImplementedError(f"[Experiments] optimizer config {json_config[OPTIMIZER][NAME]} is not defined.")
    return optimizer

def get_uncertainty_methods(json_config):
    """ Instantiate the functions associated with the chosen methods.

    Args:
        json_config (dict): Configuration extracted from the json file

    Raises:
        NotImplementedError: if an UQ name is not recognized

    Returns:
        list<dict>: list of uncertainty methods
    """
    uq_methods_list = []
    if UQ in json_config.keys():
        for uq_json in json_config[UQ]:
            uq_method = {
                NAME: uq_json[NAME],
                PRED_POS: [int(pp) for pp in uq_json[PRED_POS]],
                LABELS: uq_json[LABELS]
                }
            if "entropy" in uq_json[NAME].lower():
                uq_method[FN] = {
                    TU: total_uncertainty_entropy,
                    AU: aleatoric_uncertainty_entropy,
                    EU: epistemic_uncertainty_entropy
                }
            else:
                raise NotImplementedError(f"[Experiments] uncertainty quantification method {uq_json[NAME]} is not recognized.")
            uq_methods_list.append(uq_method)
    return uq_methods_list

def configJSON2obj(json_config):
    """
    Translate the JSON config dictionary to a dictionary with instantiated dataloaders, model, losses,
    except metrics.

    Arguments:
    ----------
        json_config: (dict)
        An experiment configuration file

    Returns:
    --------
        A configuration dict with instantiated objects. The json config is also altered during the function to provide information on some computed parameters.
    """
    obj_config = {}

    # Generating the configuration dict with objects
    ## GENERAL
    obj_config[NAME] = json_config[NAME]
    if "cuda" in json_config[DEVICE]:
        obj_config[DEVICE] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        obj_config[DEVICE] = torch.device("cpu")

    ## CHECKPOINTS
    if CHECKPOINT_FREQ in json_config.keys():
        assert isinstance(json_config[CHECKPOINT_FREQ], int) and json_config[CHECKPOINT_FREQ] > 0, f"[Experiments] checkpoint frequency {json_config[CHECKPOINT_FREQ]} is not valid. It should be a positive integer."
        obj_config[CHECKPOINT_FREQ] = json_config[CHECKPOINT_FREQ]
    ## NB_EPOCHS
    obj_config[NB_EPOCHS] = json_config[NB_EPOCHS]
    ## NB_INFERENCE
    obj_config[NB_INFERENCE] = json_config[NB_INFERENCE] if NB_INFERENCE in json_config.keys() else 1
    
    ## BATCH SIZE
    if isinstance(json_config[BATCH_SIZE], list) or isinstance(json_config[BATCH_SIZE], tuple):
        obj_config[BATCH_SIZE] = {
            TRAIN: json_config[BATCH_SIZE][0],
            TEST:  json_config[BATCH_SIZE][1],
        }
    elif isinstance(json_config[BATCH_SIZE], float) or isinstance(json_config[BATCH_SIZE], int):
        obj_config[BATCH_SIZE] = {
            TRAIN: json_config[BATCH_SIZE],
            TEST: json_config[BATCH_SIZE]
        }
    else:
        raise NotImplementedError(f"[Experiments] batch size config type {type(json_config[BATCH_SIZE])} is not handled.")

    ## DATASETS (external function)
    obj_config[DATASET] = {}
    obj_config[DATALOADER], obj_config[DATASET][TRAIN], obj_config[DATASET][VAL], obj_config[DATASET][TEST] = get_dataloaders(json_config)

    ## MODEL
    obj_config[MODEL] = get_model(json_config)

    ## OPTIMIZER
    obj_config[OPTIMIZER] = get_optimizer(json_config, obj_config[MODEL])

    ## LOSSES
    obj_config[TRAIN_LABELS] = {}
    obj_config[TRAIN_LABELS][LABELS] = json_config[TRAIN_LABELS][LABELS]
    obj_config[TRAIN_LABELS][WEIGHT] = json_config[TRAIN_LABELS][WEIGHT]
    obj_config[TRAIN_LABELS][PRED_POS] = json_config[TRAIN_LABELS][PRED_POS]
    obj_config[TRAIN_LABELS][LOSS] = []
    for i, (loss, args) in enumerate(zip(json_config[TRAIN_LABELS][LOSS],json_config[TRAIN_LABELS][ARGS])):
        args_copy = {k: v for k, v in args.items()}
        
        # Manage weighting
        if "weight" in args.keys():
            args_copy["weight"] = torch.Tensor(args["weight"]).to(obj_config[DEVICE])
        elif "weighted" in  loss.lower():
            label_name = obj_config[TRAIN_LABELS][LABELS][i]
            y_train = np.array([sample[1][label_name] for sample in obj_config[DATASET][TRAIN]])
            weights = compute_class_weight(class_weight='balanced', classes=np.sort(np.unique(y_train)), y=y_train)

            args_copy["weight"] = torch.Tensor(weights).to(obj_config[DEVICE]) 
            json_config[TRAIN_LABELS][ARGS][i]["weight"] = [float(w) for w in weights] # Saving weights to json config
            print(f"[experiments] Weighted loss ({loss.lower()}), associated weights {weights}")
            
        # Implement each loss
        if "mse" in loss.lower():
            obj_config[TRAIN_LABELS][LOSS].append(torch.nn.MSELoss(**args))
        elif "gce" in loss.lower():
            if "weight" in args_copy.keys():
                args_copy["class_weights"] = args_copy["weight"]
                del args_copy["weight"]
            obj_config[TRAIN_LABELS][LOSS].append(GeneralizedCrossEntropyLoss(**args_copy))
        elif "ce" in loss.lower(): #! has to be after gce, rce and sce
            obj_config[TRAIN_LABELS][LOSS].append(torch.nn.CrossEntropyLoss(**args_copy))
        elif "mae" in loss.lower():
            # Personal WeightedMAE loss to enable weighting (+ adapted to classification)
            obj_config[TRAIN_LABELS][LOSS].append(WeightedMAELoss(**args_copy))
        elif "geojsd" in loss.lower():
            if "weight" in args_copy.keys():
                args_copy["class_weights"] = args_copy["weight"]
                del args_copy["weight"]
            obj_config[TRAIN_LABELS][LOSS].append(GeoJSDLoss(**args_copy))
        else:
            raise NotImplementedError(f"[Experiments] Loss config {loss} is not implemented.")

    ## Uncertainty quantification (UQ)
    obj_config[UQ] = get_uncertainty_methods(json_config)

    ## Evaluation
    obj_config[EVALUATION] = json_config[EVALUATION]
    
    ## Time stamps on json config
    json_config[INFO] = {}
    json_config[INFO]["date"] = datetime.datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    json_config[INFO]["git_commit"] = sha
    json_config[INFO]["git_branch"] = repo.active_branch.name
    json_config[INFO]["git_remote"] = repo.remotes.origin.url

    return obj_config

def create_run_folder(exp_path, run_name = None):
    """
    Create a folder with the name run_name (or run-i) in the experiment folder exp_name.

    Arguments:
    ----------
    exp_name: str or Path
    path to the experiment folder, created if this folder does not exist

    run_name: str
    Name for the run

    Returns:
    ----------
    str:Path to rhe run folder
    """

    exp_path = Path(exp_path)
    if not exp_path.exists():
        os.mkdir(exp_path)

    i = 0
    if run_name is None:
        run_name = f"run-{i}"
        while (exp_path/run_name).exists():
            run_name = f"run-{i}"
            i += 1

    output_folder = exp_path/run_name
    os.mkdir(output_folder)

    return output_folder

###############################################################################
###############################################################################

if __name__ == "__main__":
    print("""See corresponding scripts:
          - train_exp to train from an experiment number
          - train_config to train from a json configuration file
          """)