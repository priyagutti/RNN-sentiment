# src/train.py
"""
Very simple training script for IMDb sentiment classification (CPU only).

What this script does:
1) Loads the IMDb dataset from the folder layout (data/aclImdb/...)
2) Builds an RNN/LSTM/BiLSTM model
3) Trains for a few epochs, prints validation metrics each epoch
4) Evaluates on the test set
5) Saves:
   - the trained model -> results/model_<rnn>_seq<L>.pth
   - loss curve plot   -> results/plots/loss_<rnn>_seq<L>.png
   - per-epoch loss CSV-> results/loss_<rnn>_seq<L>.csv
   - metrics row       -> results/metrics.csv
"""

import os
import csv
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score

from src.preprocess import create_dataloaders
from src.models import SentimentRNN
from src.utils import save_metrics_to_csv, plot_loss_curve


# -------------- Reproducibility --------------
def set_seed(seed=42):
    """Fix random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seed set to {seed}")


# -------------- Device (CPU only) --------------
def get_device():
    print("Using CPU only")
    return torch.device("cpu")


def main():
    # -----------------------------
    # settings (edit these to run different experiments)
    # -----------------------------
    data_dir = "data/aclImdb"   # path to your IMDb folder
    seq_len = 50               # try 25, 50, or 100
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.001

    rnn_type = "lstm"           # "rnn", "lstm", or "bilstm"
    activation = "relu"         # "relu", "tanh", or "sigmoid"
    optimizer_name = "adam"     # "adam", "sgd", or "rmsprop"
    use_grad_clip = False       # True or False
    set_seed(42)
    device = get_device()

    # -----------------------------
    # 3) Load data
    # -----------------------------
    print("Loading and preprocessing IMDb data...")
    (
        train_loader,
        val_loader,
        test_loader,
        stats,
        word2idx,
        idx2word
    ) = create_dataloaders(
        data_dir=data_dir,
        seq_len=seq_len,
        batch_size=batch_size,
        val_ratio=0.1,   # 10% of train used as validation
    )

    print("\n Data loaded!")

    # -----------------------------
    # 4) Create the model
    # -----------------------------
    vocab_size = len(word2idx)
    model = SentimentRNN(
        vocab_size=vocab_size,
        embedding_dim=100,   
        hidden_size=64,      
        num_layers=2,        
        rnn_type=rnn_type,
        dropout=0.5,
        activation=activation,
    ).to(device)

    print(model, "\n")
    criterion = nn.BCELoss()

    # Choose optimizer
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unknown optimizer: " + optimizer_name)

    # -----------------------------
    # 5) Training loop
    # -----------------------------
    print("🚀 Starting training...")
    train_losses_all = []
    val_losses_all = []
    epoch_times = []

    for epoch in range(num_epochs):
        t0 = time.perf_counter()

        model.train()  # training mode (dropout on)
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)              # (batch_size,)
            loss = criterion(outputs, labels)    # scalar

            loss.backward()

            # Gradient clipping 
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

            # Print every 100 batches to see progress
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        train_losses_all.append(avg_train_loss)

        # ---- Validation ----
        model.eval()  
        val_loss = 0.0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(outputs.cpu().numpy())

        all_val_labels = np.array(all_val_labels)
        all_val_preds = np.array(all_val_preds)
        val_loss_avg = val_loss / len(val_loader)
        val_losses_all.append(val_loss_avg)

        # threshold at 0.5
        val_preds_binary = (all_val_preds >= 0.5).astype(int)
        val_accuracy = accuracy_score(all_val_labels, val_preds_binary)
        val_f1 = f1_score(all_val_labels, val_preds_binary, average="macro")

        epoch_time = time.perf_counter() - t0
        epoch_times.append(epoch_time)

        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {val_loss_avg:.4f}")
        print(f"Val Acc:    {val_accuracy:.4f}")
        print(f"Val F1:     {val_f1:.4f}")
        print(f"Epoch Time: {epoch_time:.2f} seconds\n")

    # -----------------------------
    # 6) Final evaluation on test set
    # -----------------------------
    print("📊 Evaluating on test set...")
    model.eval()
    all_test_labels = []
    all_test_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(outputs.cpu().numpy())

    all_test_labels = np.array(all_test_labels)
    all_test_preds = np.array(all_test_preds)
    test_preds_binary = (all_test_preds >= 0.5).astype(int)

    test_accuracy = accuracy_score(all_test_labels, test_preds_binary)
    test_f1 = f1_score(all_test_labels, test_preds_binary, average="macro")

    # -----------------------------
    # 7) Save artifacts (model, loss curves, metrics)
    # -----------------------------
    os.makedirs("results/plots", exist_ok=True)

    # 7a) Save model
    model_path = f"results/model_{rnn_type}_seq{seq_len}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Saved model to {model_path}")

    # 7b) Save loss curve plot
    plot_path = f"results/plots/loss_{rnn_type}_seq{seq_len}.png"
    plot_loss_curve(train_losses_all, val_losses_all, save_path=plot_path)

    # 7c) Save per-epoch loss CSV
    loss_csv_path = f"results/loss_{rnn_type}_seq{seq_len}.csv"
    with open(loss_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])
        for i, (tr, va) in enumerate(zip(train_losses_all, val_losses_all), start=1):
            w.writerow([i, tr, va])
    print(f" Saved per-epoch losses to {loss_csv_path}")

    # 7d) Append a row to metrics.csv
    avg_epoch_time = float(np.mean(epoch_times)) if len(epoch_times) > 0 else 0.0
    save_metrics_to_csv(
        "results/metrics.csv",
        {
            "Model": rnn_type,
            "Activation": activation,
            "Optimizer": optimizer_name,
            "SeqLength": seq_len,
            "GradClip": use_grad_clip,
            "Accuracy": round(test_accuracy, 4),
            "F1": round(test_f1, 4),
            "AvgEpochTimeSec": round(avg_epoch_time, 2),
        },
    )

    # 7e) Print final test results
    print("\n===== Test Results =====")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print("=========================")


if __name__ == "__main__":
    main()
