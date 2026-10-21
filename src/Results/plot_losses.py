
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.Utils.vocabulary import TEST, TOTAL, TRAIN, VAL


def plot_losses(folders, loss_label = "Mean loss", title = "Experiments losses", show_legend = True, legends = None, detailed = False, same_color = True):
    """
    Plot the losses from each folder/losses.json files in the folders list.
    Arguments:
    ----------
        folders: list of str or Path
        The list of folders frow which metrics to take

        loss_label: str
        Loss label for the y-axis
    """
    # plt.figure()
    for i, folder in enumerate(folders):
        folder = Path(folder)
        assert folder.exists(), f"Folder {folder} does not exist."

        with (folder/"losses.json").open("r") as f:
            losses = json.load(f)
        
        if legends is not None:
            label = legends[i]
        else:
            label = f"{folder.parent.stem}-{folder.stem}"
        if(type(losses[TRAIN]) == dict):
            for loss_key in losses[TRAIN].keys():
                if loss_key == TOTAL:
                    if i == 0 or not(same_color):
                        figTot = plt.plot(losses[TRAIN][loss_key], label = f"{label} train")
                    else:
                        figTot = plt.plot(losses[TRAIN][loss_key], label = f"{label} train", color=color)
                    if i == 0:
                        color = figTot[0].get_color()
                    else :
                        color = figTot[0].get_color()
                    if VAL in losses.keys():
                        plt.plot(losses[VAL][TOTAL], "--", color = color, label = f"validation")
                    elif TEST in losses.keys():
                        plt.plot(losses[TEST][TOTAL], "--", color = color, label = f"test")
                else:
                    if detailed:
                        figDet = plt.plot(losses[TRAIN][loss_key], "-", label = f"{loss_key} train")
                        if i == 0:
                            colorDet = figDet[0].get_color()
                        else :
                            colorDet = figDet[0].get_color()
                        if VAL in losses.keys():
                            plt.plot(losses[VAL][loss_key], "--", color = colorDet, label = f"{loss_key} test")
                        elif TEST in losses.keys():
                            plt.plot(losses[TEST][loss_key], "--", color = colorDet, label = f"{loss_key} test")
                            
        else:
            fig = plt.plot(losses[TRAIN], label = f"{label} train")
            if VAL in losses.keys():
                plt.plot(losses[VAL], "--", color = fig[0].get_color(), label = f"validation")
    plt.xlabel("Epochs")
    plt.ylabel(loss_label)
    plt.title(title)
    if show_legend:
        plt.legend(loc='upper right')
    # plt.show()
###############################################################################
###############################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiments test help")
    parser.add_argument("-p",  "--parent_folder", type=str, required=True,
                        help="Path to the experiment folder")
    parser.add_argument("-f",  "--exp_folders", type=str, nargs="+", required=True,
                        help="Path to the experiments folder from the root folder")
    parser.add_argument("-l",  "--loss", type=str, required=False, default="Mean loss",
                    help="Loss legend")
    parser.add_argument("-t",  "--title", type=str, required=False, default="Experiment losses",
                    help="Graph title")
    parser.add_argument("-nl",  "--no_legend", action="store_false", required=False, default=True,
                    help="Show legend")
    parser.add_argument("-ll",  "--legends",  type=str, nargs="+", required=False, default=None,
                    help="Legend labels")
    parser.add_argument("-d",  "--detailed", action="store_true", required=False, default=False,
                help="Show detailed loss")

    args = parser.parse_args()
    parent_folder = Path(args.parent_folder)
    exp_folders = args.exp_folders
    loss_label = args.loss
    title = args.title
    show_legend = args.no_legend
    folders = [parent_folder/exp_folder for exp_folder in exp_folders if (parent_folder/exp_folder).is_dir()]

    plot_losses(folders, loss_label, title=title, show_legend = show_legend, legends = args.legends, detailed = args.detailed)

        