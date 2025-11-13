# src/utils.py

"""
Helper functions for the IMDb sentiment project.
This version uses CPU only (no GPU or CUDA).
"""
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import csv
import os


def set_seed(seed=42):
    """
    Make the results reproducible by setting random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"✅ Random seed set to {seed}")


def compute_metrics(y_true, y_pred):
    """
    Calculate accuracy and F1 score for binary classification.

    Args:
        y_true: list or numpy array of true labels (0 or 1)
        y_pred: list or numpy array of predicted probabilities (0–1)
    Returns:
        acc, f1
    """
    y_pred_binary = (y_pred >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary, average="macro")
    return acc, f1


def plot_loss_curve(train_losses, val_losses, save_path=None):
    """
    Draws a simple line plot for training and validation loss.
    """
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"📈 Saved loss plot to {save_path}")
    else:
        plt.show()

    plt.close()


def save_metrics_to_csv(file_path, data_dict):
    """
    Saves experiment results to a CSV file.

    Example:
        save_metrics_to_csv("results/metrics.csv", {
            "Model": "LSTM",
            "Accuracy": 0.87,
            "F1": 0.85,
            "Seq Length": 50
        })
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)

    print(f"💾 Saved results to {file_path}")
