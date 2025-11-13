# src/preprocess.py

"""
Preprocessing the IMDb folder dataset for sentiment classification.

This version works with the original ACL IMDb layout:

data/aclImdb/
    train/
        pos/*.txt
        neg/*.txt
    test/
        pos/*.txt
        neg/*.txt

Each review is a .txt file.
"""

# ---------- Imports ----------
import os
import re
import random
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize

# ---------- Global Settings ----------
MAX_VOCAB_SIZE = 10_000   # keep only the 10,000 most common words
PAD_IDX = 0               # ID for padding token
UNK_IDX = 1               # ID for unknown words


# ---------- Step 1: Clean text ----------
def clean_text(text: str) -> str:
    """
    Make text lowercase, remove punctuation and extra spaces.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # keep only letters, digits, spaces
    text = re.sub(r"\s+", " ", text).strip()   # collapse multiple spaces
    return text


# ---------- Step 2: Build vocabulary ----------
def build_vocab(texts: List[str], max_size: int = MAX_VOCAB_SIZE):
    """
    Count all words in the training texts and keep the most common ones.
    Assign each word a unique ID.
    """
    word_counter = Counter()

    for t in texts:
        tokens = word_tokenize(clean_text(t))
        word_counter.update(tokens)

    # reserve 2 spots for PAD and UNK
    most_common = word_counter.most_common(max_size - 2)

    word2idx = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    for i, (word, _) in enumerate(most_common, start=2):
        word2idx[word] = i

    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word


# ---------- Step 3: Text -> sequence of IDs ----------
def text_to_sequence(text: str, word2idx) -> List[int]:
    tokens = word_tokenize(clean_text(text))
    return [word2idx.get(tok, UNK_IDX) for tok in tokens]


# ---------- Step 4: Pad or cut sequences ----------
def pad_sequence(seq: List[int], max_len: int) -> List[int]:
    """
    Make all sequences the same length (max_len).
    """
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [PAD_IDX] * (max_len - len(seq))


# ---------- Step 5: PyTorch Dataset ----------
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, word2idx, seq_len: int):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = text_to_sequence(self.texts[idx], self.word2idx)
        seq = pad_sequence(seq, self.seq_len)

        x = torch.tensor(seq, dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# ---------- Step 6: Load IMDb from folders ----------
def load_imdb_from_files(data_dir: str) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load the IMDb dataset from the folder structure.

    data_dir should be "data/aclImdb".
    Train and test are already predefined: 25k train, 25k test.
    """

    def load_split(split: str):
        split_texts = []
        split_labels = []

        # pos -> label 1, neg -> label 0
        for label_name, label_value in [("pos", 1), ("neg", 0)]:
            folder = os.path.join(data_dir, split, label_name)
            for fname in os.listdir(folder):
                if not fname.endswith(".txt"):
                    continue
                path = os.path.join(folder, fname)
                with open(path, encoding="utf-8") as f:
                    split_texts.append(f.read())
                    split_labels.append(label_value)
        return split_texts, split_labels

    train_texts, train_labels = load_split("train")
    test_texts, test_labels = load_split("test")
    return train_texts, train_labels, test_texts, test_labels


# ---------- Step 7: Create DataLoaders ----------
def create_dataloaders(
    data_dir: str,
    seq_len: int,
    batch_size: int = 32,
    val_ratio: float = 0.1,
):
    """
    Create train, validation, and test DataLoaders.

    - Uses the predefined train/test split from the folder.
    - Further splits train into train/val using val_ratio.
    """
    # 1. Load raw texts and labels from disk
    train_texts, train_labels, test_texts, test_labels = load_imdb_from_files(data_dir)

    # 2. Build vocabulary from TRAIN texts only
    word2idx, idx2word = build_vocab(train_texts, max_size=MAX_VOCAB_SIZE)
    vocab_size = len(word2idx)

    # 3. Split training into train/val
    indices = list(range(len(train_texts)))
    random.shuffle(indices)
    split = int((1 - val_ratio) * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_texts_split = [train_texts[i] for i in train_idx]
    train_labels_split = [train_labels[i] for i in train_idx]
    val_texts_split = [train_texts[i] for i in val_idx]
    val_labels_split = [train_labels[i] for i in val_idx]

    # 4. Create Dataset objects
    train_ds = IMDBDataset(train_texts_split, train_labels_split, word2idx, seq_len)
    val_ds = IMDBDataset(val_texts_split, val_labels_split, word2idx, seq_len)
    test_ds = IMDBDataset(test_texts, test_labels, word2idx, seq_len)

    # 5. Wrap them in DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 6. Some simple stats for your report
    avg_train_len = np.mean([len(text_to_sequence(t, word2idx)) for t in train_texts])
    print(f"Vocabulary size: {vocab_size}")
    print(f"Average training review length: {avg_train_len:.2f} words")

    stats = {"vocab_size": vocab_size, "avg_train_len": avg_train_len}
    return train_loader, val_loader, test_loader, stats, word2idx, idx2word
