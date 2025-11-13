# src/evaluate.py

"""
This script checks how good your trained model is on the IMDb test data.
It runs only on CPU
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from src.models import SentimentRNN
from src.preprocess import create_dataloaders


def evaluate(model, data_loader):
    """
    Go through the test data and compute accuracy and F1-score.
    """
    model.eval()  # turn off training mode (no dropout)
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # we don't need gradients for testing
        for inputs, labels in data_loader:
            # Send data to CPU (everything stays on CPU)
            outputs = model(inputs)

            all_labels.extend(labels.numpy())
            all_predictions.extend(outputs.numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Convert probabilities (0–1) into 0 or 1
    preds_binary = (all_predictions >= 0.5).astype(int)

    acc = accuracy_score(all_labels, preds_binary)
    f1 = f1_score(all_labels, preds_binary, average="macro")
    return acc, f1


def main():
    # -----------------------------
    # Step 1: Settings (match training)
    # -----------------------------
    data_dir = "data/aclImdb"           # where IMDb data is stored
    seq_len = 50                        # same as training
    batch_size = 32
    model_path = "results/sentiment_model.pth"  # saved model file

    embedding_dim = 100
    hidden_size = 64
    num_layers = 2
    rnn_type = "lstm"      # "rnn", "lstm", or "bilstm"
    activation = "relu"

    print("⚙️ Running on CPU only...")

    # -----------------------------
    # Step 2: Load dataset
    # -----------------------------
    print("Loading IMDb test data...")
    _, _, test_loader, stats, word2idx, idx2word = create_dataloaders(
        data_dir=data_dir,
        seq_len=seq_len,
        batch_size=batch_size,
    )
    print(f"Test data ready! Number of batches: {len(test_loader)}")
    print()

    # -----------------------------
    # Step 3: Load model
    # -----------------------------
    print(f" Loading model from {model_path} ...")
    model = SentimentRNN(
        vocab_size=len(word2idx),
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        rnn_type=rnn_type,
        activation=activation,
    )

    # Load model weights (always map to CPU)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    print("Model loaded successfully!\n")

    # -----------------------------
    # Step 4: Evaluate
    # -----------------------------
    print("Evaluating model on test set (CPU)... please wait...")
    acc, f1 = evaluate(model, test_loader)

    print("\n===== FINAL TEST RESULTS =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("===============================")


if __name__ == "__main__":
    main()
