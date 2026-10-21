
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from src.Utils.vocabulary import ALL


def parse_metrics(folders_dict):
    """
    Gather the metrics in the different run folders of the folder_dict.

    Args:
    ---------
    folders_dict: dict {string or int or float: list of (string or Path)}
        Dictionary with a name for an experiment as a key, and a list of run paths as a value

    Returns:
    ---------
    dict: A dictionary with key a split and value a dictionary with key a metric name and value a list with a list of float for each experiment

    list: list of the metrics names
    """
    metrics = {}
    metric_list = []
    exps = list(folders_dict.keys())
    for i, exp in enumerate(exps):
        for folder in folders_dict[exp]:
            folder = Path(folder)
            assert folder.exists(), f"Folder {folder} does not exist."

            with (folder/"metric.json").open("r") as f:
                metrics_i = json.load(f)
            
            # Gathering metrics in lists
            for split in metrics_i.keys():
                if split not in metrics.keys():
                    metrics[split]  = {}
                for metric in metrics_i[split].keys():
                    if metric.lower() not in metrics[split].keys():
                        metrics[split][metric.lower()] = [[] for exp in exps]
                        metric_list.append(metric.lower())
                    metrics[split][metric.lower()][i].append(metrics_i[split][metric])
    return metrics, sorted(list(set(metric_list)))

def plot_metrics_stats(folders_dict, title='', filter_metric = None, filter_split = None, figsize = None, logscale = False, zeroline = False):
    """
    Plot the statistics (matplotlib boxplot) of experiments.

    Arguments:
    ----------
        folders_dict: dict
        A dictionary with a chosen experiment name as a key and corresponding folders as the values

        title: str
        Global title for the plot

        filter_metric: list of str, default None
        A list of metrics to plot, among the available ones
    """
    metrics, metric_list = parse_metrics(folders_dict)
    exps = list(folders_dict.keys())
    # Filter metrics
    if filter_metric is None:
        filter_metric = metric_list
    else:
        filter_metric = [metric.lower() for metric in filter_metric]
        filter_metric = sorted(list(set(filter_metric).intersection(set(metric_list))))
    # Filter splits
    if filter_split is None:
        filter_split = list(metrics.keys())

    # Plotting metrics
    if type(exps[0]) == str:
        x = [i for i in range(1, len(exps) + 1)]
        ticks = exps
        xmin = 0.5
        xmax = len(exps) + 0.5
        width = 0.5
    else :
        if logscale:
            x = np.log(exps)
        else:
            x = exps
        ticks = exps
        xmin = min(x) - 0.1 * (max(x) - min(x))
        xmax = max(x) + 0.1 * (max(x) - min(x))
        width = 0.15 * (max(exps) - min(exps))

    fig, axs = plt.subplots(len(filter_metric), len(filter_split), sharey="row", figsize=figsize if figsize is not None else (5, 12))
    fig.tight_layout(h_pad = 4, w_pad = 2)
    metrics_stat = {}
    for i, metric in enumerate(filter_metric):
        metrics_stat[metric] = {}
        for j, split in enumerate(filter_split):
            ax = axs[i, j] if len(filter_split) > 1 else axs[i]
            metrics_stat[metric][split] = {}
            data = metrics[split][metric]
            ax.boxplot(data, positions = x, widths = width)
            if zeroline:
                ax.plot([xmin, xmax], [0] * 2, '--', color = "lightgray")
            ax.set_xticks(x, ticks, rotation=10, ha="right")
            ax.set_xlim(xmin, xmax)
            ax.set_title(f"{metric} on {split} data")
            ax.grid('minor')
            if j == 0:
                if metric.lower() == "mse":
                    # ax.set_ylim(0, 0.1)
                    ax.set_ylabel("Mean square error")
                elif "silhouette" in metric.lower():
                    # ax.set_ylim(-0.4, 0.4)
                    ax.set_ylabel("Silhouette score")
                elif "mcc" in metric.lower():
                    # ax.set_ylim(0.6, 1)
                    ax.set_ylabel("MCC")

            # Computing mean and standard deviation
            for k, exp in enumerate(exps):
                metrics_stat[metric][split][exp] = np.mean(data[k]), np.std(data[k]), np.median(data[k])
    fig.suptitle(title)
    plt.subplots_adjust(top=0.92)
    plt.show()

    return metrics_stat

def print_metrics_stat(metrics_stat, split = ALL):
    """Print a metric_stat dictionary in a readable form of table

    Args:
        metrics_stat (dict): _description_
        split (str, optional): data split name. Defaults to ALL.
    """
    print(f"Experiments statistics (mean, std, median)")
    table_all = []
    metrics = list(metrics_stat.keys())
    exps = list(metrics_stat[metrics[0]][split].keys())
    table_all.append([None,] + exps)
    for metric_type in metrics_stat.keys():
        exp_metrics = metrics_stat[metric_type][split]
        table_all.append([metric_type, ]+ [val for key, val in exp_metrics.items()])
    print(tabulate(table_all))


###############################################################################
###############################################################################

if __name__ == "__main__":
    print("""
          See corresponding notebooks.
          """)