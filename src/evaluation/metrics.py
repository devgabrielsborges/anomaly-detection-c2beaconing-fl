"""Evaluation metrics and visualization for imbalanced classification."""

import logging
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    pos_label: int = 1,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for binary classification.

    Particularly useful for imbalanced datasets like anomaly detection.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities for positive class (optional)
        pos_label: Label of positive class

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(
        y_true, y_pred, pos_label=pos_label, zero_division=0
    )
    metrics["recall"] = recall_score(
        y_true, y_pred, pos_label=pos_label, zero_division=0
    )
    metrics["f1"] = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)

    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    metrics["mcc"] = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    metrics["balanced_accuracy"] = (metrics["recall"] + metrics["specificity"]) / 2

    # G-mean (geometric mean of recall and specificity)
    metrics["g_mean"] = np.sqrt(metrics["recall"] * metrics["specificity"])

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

            # Precision-Recall AUC (better for imbalanced data)
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            metrics["pr_auc"] = auc(recall, precision)
        except ValueError as e:
            logger.warning(f"Could not compute AUC metrics: {e}")
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0

    # Support (number of samples)
    metrics["support_positive"] = int(np.sum(y_true == pos_label))
    metrics["support_negative"] = int(np.sum(y_true != pos_label))
    metrics["support_total"] = len(y_true)

    # Imbalance ratio
    if metrics["support_negative"] > 0:
        metrics["imbalance_ratio"] = (
            metrics["support_negative"] / metrics["support_positive"]
        )
    else:
        metrics["imbalance_ratio"] = 0.0

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    normalize: bool = False,
    figsize: tuple = (8, 6),
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Whether to normalize values
        figsize: Figure size
        cmap: Color map

    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    if class_names is None:
        class_names = ["Background", "Anomaly"]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Plot Precision-Recall curve.

    This is particularly useful for imbalanced datasets.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    # Baseline (random classifier performance)
    baseline = np.sum(y_true) / len(y_true)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        recall,
        precision,
        color="darkorange",
        lw=2,
        label=f"PR curve (AUC = {pr_auc:.2f})",
    )
    ax.axhline(
        y=baseline,
        color="navy",
        lw=2,
        linestyle="--",
        label=f"Random (baseline = {baseline:.2f})",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
