import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_multiclass_calibration(y_true, softmax_probs, n_bins=10, class_names=None, ax=None, class_colors=None, strategy="uniform"):
    """
    Plot calibration curves for multi-class softmax predictions.
    
    Parameters
    ----------
    y_true : array-like of shape (N,)
        True integer labels.
    softmax_probs : array-like of shape (N, C)
        Softmax probabilities for C classes.
    n_bins : int
        Number of calibration bins.
    class_names : list of str
        Optional class names for plotting.
    ax : matplotlib ax
        Optional ax if plotted in a global subplot. If None, the whole plot is plotted
    class_colors : list of colors
        Optional list of colors for each class curve.
    strategy : str
        Strategy for binning the predicted probabilities. "uniform" or "quantile".
    """
    num_classes = softmax_probs.shape[1]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    for c in range(num_classes):

        y_true_binary = (y_true == c).astype(int)
        prob_true, prob_pred = calibration_curve(
            y_true_binary, softmax_probs[:, c],
            n_bins=n_bins, strategy=strategy
        )

        ax.plot(prob_pred, prob_true, 
                "s-", 
                label=class_names[c], 
                color=class_colors[c] if class_colors is not None else None,
                # marker=class_markers[c] if class_markers is not None else None,
                )
    ax.plot([0, 1], [0, 1], "k--", label="Calibrated model")
    ax.set_title(f"Calibration curve")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")

    if ax is None:
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()
    