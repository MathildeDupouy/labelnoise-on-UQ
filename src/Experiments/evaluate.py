import json
from csv import Error
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

from src.Utils.tools import update_saved_dict
from src.Utils.vocabulary import (
    ALL,
    ARGS,
    BINARY_TARGET,
    DATASET,
    EVALUATION,
    HARD_TARGET,
    LABELS,
    METRIC,
    MULTI_TARGET,
    NB_INFERENCE,
    PREDICTED,
    TEST,
    TRAIN,
    VAL,
)

# Handled metrics:
# classification: mcc, acc, f1, top-2 acc (XXX in class, pos, hard-target, multi-target, binary-target), confusion matrix (lines are the true class, columns are the predicted class, values are the number of samples)

CLASSIFICATION_METRICS = ["mcc class", "mcc pos", "mcc hard-target", "mcc multi-target", "mcc binary-target",
                          "acc class", "acc pos", "acc hard-target", "acc multi-target", "acc binary-target",
                          "f1 class", "f1 pos", "f1 hard-target", "f1 multi-target", "f1 binary-target",
                          "top-2 acc class", "top-2 acc pos", "top-2 acc hard-target", "top-2 acc multi-target", "top-2 acc binary-target",
                          "confusion matrix class", "confusion matrix pos", "confusion matrix hard-target", "confusion matrix multi-target", "confusion matrix binary-target",
                          "classification report class", "classification report pos", "classification report hard-target", "classification report multi-target", "classification report binary-target",
                          "balanced acc class", "balanced acc pos", "balanced acc hard-target", "balanced acc multi-target", "balanced acc binary-target",
                          ]

def evaluate(exp_config, exp_path, predictions, representations = None, metric_path_name="metric.json"):
    """
    Evaluate an experiment based on the objects instantiated in exp_config and an experiment path.
    
    Arguments:
    ----------
        exp_config: dictionary
            Experiment configuration, generally loaded from a configuration file
        exp_path: str or Path
            Path to the folder to store the generated files
        predictions: dictionary
            The predictions associated to the dataset of the configuration
    Returns:
    --------
        dictionary of metrics
    """
    exp_path = Path(exp_path)
    assert exp_path.exists(), f"Path {exp_path} doesn't exist"
    # Finding test set in predictions (when no validation, test samples are inferred during training for information)
    print(f"[evaluate_exp] Modes found in predictions: {predictions.keys()}")
    print(f"[evaluate_exp] Modes found in config DATASET: {exp_config[DATASET].keys()}")
    has_val_config = VAL in exp_config[DATASET].keys() and exp_config[DATASET][VAL] is not None and len(exp_config[DATASET][VAL]) > 0
    has_test_config = TEST in exp_config[DATASET].keys() and exp_config[DATASET][TEST] is not None and len(exp_config[DATASET][TEST]) > 0
    has_val_pred = VAL in predictions.keys()
    has_test_pred = TEST in predictions.keys()
    print(f"[evaluate_exp] has_val_config: {has_val_config}, has_test_config: {has_test_config}, has_val_pred: {has_val_pred}, has_test_pred: {has_test_pred}")
    if has_test_config and not has_val_config and has_val_pred and not has_test_pred:
        predictions[TEST] = predictions[VAL]
        print(f"[evaluate_exp] Test set not found in predictions, using validation set as test set.")
    modes = {
        mode: exp_config[DATASET][mode] 
        for mode in [TRAIN, VAL, TEST] 
        if (mode in exp_config[DATASET].keys())
        and exp_config[DATASET][mode] is not None
        and len(exp_config[DATASET][mode]) > 0 
        and (mode in predictions.keys())
        }
    
    # ------ Adapting to the number of inference
    # Prediction size is [B, D0, ...., Dp, nb inference]
    accumulate_pred = False
    if NB_INFERENCE in exp_config and exp_config[NB_INFERENCE] > 1:
        accumulate_pred = True
    
    # ------ Computing metrics
    metrics = {ALL: {}, TRAIN: {}, VAL:{}, TEST: {}}
    for idx, metric in enumerate(exp_config[EVALUATION][ALL][METRIC]):
        print(f"======Evaluating metric {metric}======")
        metric_name = metric.lower()
        args = exp_config[EVALUATION][ALL][ARGS][idx] if ARGS in exp_config[EVALUATION][ALL].keys() else None

        #--- CLASSIFICATION METRICS -----------#
        if metric.lower() in CLASSIFICATION_METRICS:
            # Getting labels
            if HARD_TARGET.lower() in metric.lower():
                label_key = HARD_TARGET
            elif MULTI_TARGET.lower() in metric.lower():
                label_key = MULTI_TARGET
            elif BINARY_TARGET.lower() in metric.lower():
                label_key = BINARY_TARGET
            else:
                raise NotImplementedError(f"[evaluate] Classification metric '{metric}' not implemented (classification unknown).")
            
            pred_all = []
            pred_ranking_all = []
            labels_all = []
            for mode in modes.keys():
                # Finding corresponding prediction and label
                if label_key not in predictions[mode].keys(): # hard label key not present, maybe soft one 
                    raise ValueError(f"{label_key} not in prediction keys for mode {mode} ({predictions[mode].keys()})")
                else:
                    pred = predictions[mode][label_key][PREDICTED]
                    labels = predictions[mode][label_key][LABELS]

                # Manage nb inference > 1
                if accumulate_pred:
                    pred = np.mean(pred, axis = -1)
                
                # If soft predicted
                if pred.shape != labels.shape:
                    print(f"[evaluate_exp] Metric {metric.lower()} Getting argmax of soft prediction (pred shape {pred.shape}, label shape {labels.shape})")
                    pred_ranking = np.argsort(pred, axis=1)[:, ::-1] # Get indices of predictions sorted from highest to lowest (for top-k accuracy)
                    pred = np.argmax(pred, axis = 1).squeeze() #TODO squeeze can pose problems depending on labels shape
                    print(f"[evaluate_exp] (final pred shape {pred.shape})")

                # Filtering categories
                if "filter_categories" in args.keys():
                    metric_name = metric.lower() + f"_filter-{'-'.join(map(str, args['filter_categories']))}"
                    filtered_labels = np.zeros_like(labels)
                    filtered_pred = np.zeros_like(pred)
                    for c in args["filter_categories"]:
                        filtered_labels += np.where(labels == c, 1, 0)
                        filtered_pred += np.where(pred == c, 1, 0)
                    labels = filtered_labels
                    pred = filtered_pred

                # Computing mode metric
                if "mcc" in metric.lower():
                    metrics[mode][metric_name] = matthews_corrcoef(labels, pred)
                elif "top-2 acc" in metric.lower():
                    top_2_pred = pred_ranking[:, :2].squeeze() # Get top 2 predicted classes
                    correct_top_2 = np.any(top_2_pred == labels[:, None], axis=1) # Check if true label is in top 2 predictions
                    metrics[mode][metric_name] = correct_top_2.mean() # Top-2 accuracy is the mean of correct predictions
                elif "balanced acc" in metric.lower():
                    metrics[mode][metric_name] = balanced_accuracy_score(labels, pred)
                elif "acc" in metric.lower(): #! Warning: "acc" should be after "balanced acc" to avoid being taken in "balanced acc"
                    metrics[mode][metric_name] = accuracy_score(labels, pred)
                elif "f1" in metric.lower():
                    average_arg = args["average"] if "average" in args.keys() else "binary"
                    metrics[mode][metric_name] = f1_score(labels, pred, average=average_arg)
                elif "confusion matrix" in metric.lower():
                    # Compute confusion matrix elements
                    cm = confusion_matrix(labels, pred)
                    metrics[mode][metric_name] = [[float(cm[i, j]) for j in range(cm.shape[1])] for i in range(cm.shape[0])] # To be json serializable
                elif "classification report" in metric.lower():
                    report = classification_report(labels, pred, **args)
                    metrics[mode][metric_name] = report
                else:
                    raise NotImplementedError(f"[evaluate] Metric '{metric}' not implemented.")
                pred_all.append(pred)
                pred_ranking_all.append(pred_ranking) if 'top-2 acc' in metric.lower() else None
                labels_all.append(labels)
            
            # Computing total metric
            if "mcc" in metric.lower():
                metrics[ALL][metric_name] = matthews_corrcoef(np.concatenate(labels_all, 0), np.concatenate(pred_all, 0))
            elif "top-2 acc" in metric.lower():
                pred_ranking_all = np.concatenate(pred_ranking_all, 0).squeeze()
                labels_all_concat = np.concatenate(labels_all, 0)
                correct_top_2 = np.any(pred_ranking_all[:, :2] == labels_all_concat[:, None], axis=1)
                metrics[ALL][metric_name] = correct_top_2.mean()
            elif "balanced acc" in metric.lower():
                metrics[ALL][metric_name] = balanced_accuracy_score(np.concatenate(labels_all, 0), np.concatenate(pred_all, 0))
            elif "acc" in metric.lower():
                metrics[ALL][metric_name] = accuracy_score(np.concatenate(labels_all, 0), np.concatenate(pred_all, 0))
            elif "f1" in metric.lower():
                average_arg = args["average"] if "average" in args.keys() else "binary"
                metrics[ALL][metric_name] = f1_score(np.concatenate(labels_all, 0), np.concatenate(pred_all, 0), average=average_arg)
            elif "confusion matrix" in metric.lower():
                # Compute confusion matrix elements
                cm = confusion_matrix(np.concatenate(labels_all, 0), np.concatenate(pred_all, 0))
                metrics[ALL][metric_name] = [[float(cm[i, j]) for j in range(cm.shape[1])] for i in range(cm.shape[0])] # To be json serializable
            elif "classification report" in metric.lower():
                report = classification_report(np.concatenate(labels_all, 0), np.concatenate(pred_all, 0), **args)
                metrics[ALL][metric_name] = report
            else:
                raise NotImplementedError(f"[evaluate] Metric '{metric}' not implemented.")
        else:
            raise NotImplementedError(f"[evaluate] Metric '{metric}' not implemented.")


    # Saving metrics
    metric_path = exp_path/metric_path_name
    if metric_path.exists():
        update_saved_dict(metric_path, metrics)
    else:
        with metric_path.open("w") as f:
            print(metrics)
            json.dump(metrics, f, indent=4)

    return metrics

###############################################################################
###############################################################################

if __name__ == "__main__":
    print("""
        cf. script in Scripts/evaluate_exp.py
          """)