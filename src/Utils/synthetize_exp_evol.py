# The aim of these methods is to summarize informations in experiments checkpoints into evolution curves,
# stored in the corresponding experiment, that are sufficient to plot and prevent from reading all files.
import json
import os
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
from src.Experiments.experiments import configJSON2obj
from torch.nn.functional import softmax

from src.Utils.vocabulary import (
    CHECKPOINT_FREQ,
    INDEX,
    LABELS,
    NB_EPOCHS,
    PREDICTED,
    TEST,
    TRAIN,
    TRAIN_LABELS,
)


def get_checkpoint_epochs(n_epochs, checkpoint_freq):
    """Get the list of checkpoint epochs given the total number of epochs and the checkpoint frequency."""
    return [ep for ep in range(checkpoint_freq - 1, n_epochs, checkpoint_freq)]

class ExperimentEvolution:
    """Class to synthetize the evolution of scores and accuracies for an experiment, distinguishing noisy and clean samples.
    The evolution is computed from the predictions stored in the experiment folder, and the clean labels are obtained from the evaluation prediction file.
    The results are stored in the experiment folder in a subfolder "evolution_summary" with subfolders for each mode (train, test).
    The scores are computed as the k-th largest value of the softmax output for each sample, and are stored in a dictionary with keys "score_class_k" for each class k, and values are dicts with
    """
    def __init__(self, exp_path, mode=TRAIN):
        self.exp_path = Path(exp_path)
        self.config_path = self.exp_path / "config.json"
        self.mode = mode

        self.config = None
        with open(self.config_path, "r") as f:
            config_json = json.load(f)
            self.config = configJSON2obj(config_json)
        self.target_label = self.config[TRAIN_LABELS][LABELS][0] if len(self.config[TRAIN_LABELS][LABELS]) == 1 else None
        assert self.target_label is not None, "Only one target label is supported for now."
        self.n_epochs = self.config[NB_EPOCHS]
        self.checkpoint_freq = self.config[CHECKPOINT_FREQ]
        self.checkpoint_epochs = get_checkpoint_epochs(self.n_epochs, self.checkpoint_freq)
        self.prediction_paths = [self.exp_path / f"predictions_ep{ep}.pkl" for ep in self.checkpoint_epochs]
        self.evaluation_path = self.exp_path / "predictions.pkl"
        
        # Load predictions and labels
        self.predictions = [self.load_prediction(pred_path) for pred_path in self.prediction_paths]
        self.evaluation_predictions = self.load_prediction(self.evaluation_path)
        self.clean_labels, self.noisy_labels = self.get_labels()
        self.noisy_mask = self.noisy_labels != self.clean_labels
        print(f"Ratio of noisy samples: {self.noisy_mask.sum()}/{len(self.noisy_mask)} = {self.noisy_mask.sum()/len(self.noisy_mask)}  ")
        
        # Results
        self.scores = {}        # Keys "score_class_k" for each class k, values are dicts with keys "noisy" and "clean"
        self.accuracies = {}    # Keys "accuracy" or "accuracy_with_noise", values are dicts with keys "noisy" and "clean"
        self.losses = {}        # Keys "loss"
    
    def load_prediction(self, prediction_path):
        """Load predictions from the given path and sort it by dataset index.

        Args:
            prediction_path (str or Path): The path to the prediction file.

        Returns:
            np.ndarray: The loaded and sorted predictions.
        """
        with open(prediction_path, "rb") as f:
            predictions = pkl.load(f)
        
        # Sort predictions by dataset index
        pred_indices = np.array(predictions[self.mode][INDEX])
        for key in predictions[self.mode].keys():
            if key != INDEX: # Keys are the label names
                #! UQ key is not sorted by dataset index, because not always present
                predictions[self.mode][key][PREDICTED] = np.array(predictions[self.mode][key][PREDICTED])[np.argsort(pred_indices)]
                predictions[self.mode][key][LABELS] = np.array(predictions[self.mode][key][LABELS])[np.argsort(pred_indices)]
            else:
                print(f"Sorting predictions {self.mode}/{key} by dataset index...")
                predictions[self.mode][key] = np.array(predictions[self.mode][key])[np.argsort(pred_indices)]
        return predictions
    
    def get_labels(self):
        """Get clean and noisy labels from the predictions.

        Returns:
            np.ndarray  : The clean labels.
            np.ndarray  : The noisy labels.
        """
        # Get noisy labels from the first checkpoint predictions, as one unique dataset is used for all checkpoints
        first_prediction = self.predictions[0]
        noisy_labels = np.array(first_prediction[self.mode][self.target_label][LABELS])
        # Get clean labels from the evaluation prediction file
        clean_labels = np.array(self.evaluation_predictions[self.mode][self.target_label][LABELS]).squeeze()
        return clean_labels, noisy_labels
        
            
    def get_score_evolution(self):
        """For each checkpoint, compute the scores for each class and store them in self.scores.
        Predictions go through a softmax before computing the scores, and the scores are computed as the k-th largest value of the softmax output for each sample.
        The different rank of the scores are stored with key "score_class_k", and the values are dicts with keys "noisy" and "clean" containing the scores for noisy and clean samples respectively.
        score dict:
        {
            "score_class_0": {
                "noisy": [np.array(score_k) for each checkpoint],
                "clean": [np.array(score_k) for each checkpoint]
            },
            "score_class_1": {
                "noisy": [np.array(score_k) for each checkpoint],
                "clean": [np.array(score_k) for each checkpoint]
            }, ...
        }
        """
        # Get scores
        nb_classes = self.predictions[0][self.mode][self.target_label][PREDICTED].shape[1]

        # Initialize the dictionary once
        self.scores = {
            f"score_class_{k}": {
                'noisy': [],
                'clean': []
            }
            for k in range(1, nb_classes + 1)
        }

        for prediction in self.predictions:
            pred = softmax(
                torch.from_numpy(
                    prediction[self.mode][self.target_label][PREDICTED]
                ),
                dim=1
            ).numpy()

            # Sort once: descending order
            # sorted_pred[:, 0] = highest score
            # sorted_pred[:, 1] = second highest score
            # ...
            sorted_pred = np.sort(pred, axis=1)[:, ::-1]

            # Add each ranked score to the corresponding dictionary entry
            for k in range(1, nb_classes + 1):
                score_k = sorted_pred[:, k - 1]

                self.scores[f"score_class_{k}"]['noisy'].append(
                    score_k[self.noisy_mask]
                )
                self.scores[f"score_class_{k}"]['clean'].append(
                    score_k[~self.noisy_mask]
                )

        print(len(self.scores["score_class_1"]['noisy']), len(self.scores["score_class_1"]['clean']))

    
    def get_acc_evolution(self):
        """Compute accuracy for each checkpoint, with respect to the clean labels or the noisy labels.
        
        The accuracies are stored in a dictionary with keys "accuracy" and "accuracy_with_noise", and values are dicts with keys "noisy" and "clean".
        The values are lists of numpy arrays, one for each checkpoint.

        accuracies dict:
        {
            "accuracy": {
                "noisy": [np.array(acc) for each checkpoint],
                "clean": [np.array(acc) for each checkpoint]
            },
            "accuracy_with_noise": {
                "noisy": [np.array(acc) for each checkpoint],
                "clean": [np.array(acc) for each checkpoint]
            },
        }
        """
        self.accuracies["accuracy"] = {'noisy': [], 'clean': []}
        self.accuracies["accuracy_with_noise"] = {'noisy': []}
        for prediction in self.predictions:
            pred = softmax(torch.from_numpy(prediction[self.mode][self.target_label][PREDICTED]), dim=1).numpy()
            
            # Get accuracy
            predicted_labels = np.argmax(pred, axis=1)
            acc_all = (predicted_labels == self.clean_labels)
            if self.noisy_mask.sum() > 0:
                self.accuracies["accuracy"]['noisy'].append(acc_all[self.noisy_mask].mean())
                acc_all_with_noise = (predicted_labels == self.noisy_labels)
                self.accuracies["accuracy_with_noise"]['noisy'].append(acc_all_with_noise[self.noisy_mask].mean())
            if (~self.noisy_mask).sum() > 0:
                self.accuracies["accuracy"]['clean'].append(acc_all[~self.noisy_mask].mean())
            # print(f"Accuracy: mean noisy = {self.accuracies['accuracy']['noisy']}, mean clean = {self.accuracies['accuracy']['clean']}")
            # print(f"Accuracy with noise: mean noisy = {self.accuracies['accuracy_with_noise']['noisy']}")
        print(len(self.accuracies["accuracy"]['noisy']), len(self.accuracies["accuracy"]['clean']), len(self.accuracies["accuracy_with_noise"]['noisy']))
    
    def get_loss_evolution(self):
        """Get loss evolution from experiment (already stored during training).
        """
        print("Loss evolution distinguishing noisy and clean samples is not implemented yet, only a copy of the global losses is done.")
        assert (self.exp_path/"losses.json").exists(), f"Losses file {self.exp_path/'losses.json'} not found."
        with open(self.exp_path/"losses.json", "r") as f:
            losses = json.load(f)
        assert self.mode in losses, f"Mode {self.mode} not found in losses file."
        assert "total" in losses[self.mode], f"Total loss not found in losses file for mode {self.mode}."
        self.losses["loss"] = losses[self.mode]["total"]
        
    
    def store_results(self):
        """Store the scores, accuracies and losses in the experiment folder.
        """
        results_path = self.exp_path / "evolution_summary" / self.mode
        os.makedirs(results_path, exist_ok=True)
        for key in self.scores.keys():
            with open(results_path / f"{key}.pkl", "wb") as f:
                pkl.dump(self.scores[key], f)
        for key in self.accuracies.keys():
            with open(results_path / f"{key}.pkl", "wb") as f:
                pkl.dump(self.accuracies[key], f)
        for key in self.losses.keys():
            with open(results_path / f"{key}.pkl", "wb") as f:
                pkl.dump(self.losses[key], f)
        with open(results_path / "checkpoint_epochs.pkl", "wb") as f:
            pkl.dump(self.checkpoint_epochs, f)
        print(f"Results stored in {results_path}")

def load_evol_results(exp_path, mode=TRAIN):
    """Load the scores and accuracies from the experiment folder.

    Args:
        exp_path (str or Path): The path to the experiment folder.

    Returns:
        dict: A dictionary containing the loaded scores and accuracies.
    """
    results_path = Path(exp_path) / "evolution_summary" / mode
    if not results_path.exists():
        raise FileNotFoundError(f"Results folder {results_path} not found.")

    scores = {}
    accuracies = {}
    losses = {}
    for file in results_path.iterdir():
        if file.name.startswith("score_class_"):
            with open(file, "rb") as f:
                scores[file.stem] = pkl.load(f)
        elif file.name.startswith("accuracy"):
            with open(file, "rb") as f:
                accuracies[file.stem] = pkl.load(f)
        elif file.name.startswith("loss"):
            with open(file, "rb") as f:
                losses[file.stem] = pkl.load(f)
        elif file.name.startswith("checkpoint_epochs"):
            with open(file, "rb") as f:
                checkpoint_epochs = pkl.load(f)
        else:
            print(f"Unknown file {file} in results folder {results_path}, ignoring.")
    return scores, accuracies, losses, checkpoint_epochs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Synthetize exp evolution help")
    parser.add_argument("-e",  "--exp_paths", type=str, nargs="+", required=True,
                        help="Path to the experiments to synthetize")

    args = parser.parse_args()
    for exp_path in args.exp_paths:
        for mode in [TRAIN, TEST]:
            evolution = ExperimentEvolution(exp_path, mode=mode)
            evolution.get_score_evolution()
            evolution.get_acc_evolution()
            evolution.get_loss_evolution()
            evolution.store_results()
