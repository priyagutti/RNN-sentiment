# src/models.py

"""
This file defines the RNN models for sentiment classification.

It includes:
- A basic RNN
- An LSTM
- A Bidirectional LSTM

All of them share the same structure:
    Embedding -> RNN/LSTM -> Fully connected layer -> Sigmoid output
"""

import torch
import torch.nn as nn


class SentimentRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_size=64,
        num_layers=2,
        rnn_type="rnn",      
        dropout=0.5,
        activation="relu",   
    ):
        super(SentimentRNN, self).__init__()

        # ----- 1. Embedding layer -----
        # Converts word IDs into dense vectors of size embedding_dim.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # ----- 2. Choose the RNN type -----
        self.rnn_type = rnn_type.lower()

        # Bidirectional .
        self.bidirectional = self.rnn_type == "bilstm"
        num_directions = 2 if self.bidirectional else 1

        # Choose which RNN to use
        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=False,
            )
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=False,
            )
        elif self.rnn_type == "bilstm":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )
        else:
            raise ValueError("rnn_type must be 'rnn', 'lstm', or 'bilstm'")

        # ----- 3. Activation function -----
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("activation must be 'relu', 'tanh', or 'sigmoid'")

        # ----- 4. Fully connected output layer -----
        # For bidirectional, we double the hidden size since it concatenates both directions.
        self.fc = nn.Linear(hidden_size * num_directions, 1)

        # ----- 5. Final sigmoid for binary classification -----
        self.output = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the network.

        Input:
            x: Tensor of shape (batch_size, seq_len)
        Output:
            out: Tensor of shape (batch_size,), values between 0 and 1
        """
        # Step 1: Convert word IDs into embeddings
        embedded = self.embedding(x)

        # Step 2: Run through the RNN/LSTM
        if self.rnn_type in ["lstm", "bilstm"]:
            rnn_out, (h_n, c_n) = self.rnn(embedded)
        else:
            rnn_out, h_n = self.rnn(embedded)

        # Step 3: Use the last hidden state (or last time step)
        # rnn_out shape: (batch_size, seq_len, hidden_size * num_directions)
        last_hidden = rnn_out[:, -1, :]

        # Step 4: Apply chosen activation
        activated = self.activation(last_hidden)

        # Step 5: Pass through the fully connected layer
        logits = self.fc(activated)

        # Step 6: Apply sigmoid to get probabilities
        probs = self.output(logits).squeeze(1)

        return probs
