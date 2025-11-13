# RNN Sentiment Classification

A PyTorch implementation comparing RNN architectures (RNN, LSTM, BiLSTM) for binary sentiment classification on the IMDb movie review dataset.

## Setup Instructions

### Python Version

- **Python 3.8+** (tested with Python 3.10+)

### Dependencies Installation

1. **Navigate to the project directory:**
   ```bash
   cd rnn-sentiment
   ```

2. **Install required packages:**
   ```bash
   pip install -r src/requirements.txt
   ```

3. **Verify NLTK data is available** (required for tokenization):
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

### Required Dependencies

- **PyTorch** (2.2.0) - Deep learning framework
- **NLTK** (3.9.1) - Natural language processing toolkit
- **scikit-learn** (1.5.2) - Machine learning metrics
- **pandas** (2.2.3) - Data manipulation
- **matplotlib** (3.9.2) - Visualization
- **NumPy** (1.26.4) - Numerical computing
- **tqdm** (4.66.5) - Progress bars

## Data Preparation

The project uses the **IMDb Large Movie Review Dataset** from the `data/aclImdb/` directory:

- **Training**: 25,000 labeled reviews (12,500 positive, 12,500 negative)
- **Test**: 25,000 labeled reviews (12,500 positive, 12,500 negative)

The dataset is expected to be in the following structure:
```
data/aclImdb/
├── train/
│   ├── pos/          (positive reviews)
│   ├── neg/          (negative reviews)
│   └── unsup/        (unsupervised reviews)
└── test/
    ├── pos/          (positive reviews)
    └── neg/          (negative reviews)
```

**Note**: The dataset is already included in this repository.

## How to Run

### Training

To train a sentiment classification model:

```bash
python -m src.train
```

**Configuration options** (edit in `src/train.py` lines ~45-60):

- `data_dir`: Path to the IMDb dataset (default: `"data/aclImdb"`)
- `seq_len`: Sequence length for padding/truncating (25, 50, or 100; default: 50)
- `batch_size`: Batch size for training (default: 32)
- `num_epochs`: Number of training epochs (default: 5)
- `learning_rate`: Learning rate for optimizer (default: 0.001)
- `rnn_type`: Model architecture - `"rnn"`, `"lstm"`, or `"bilstm"` (default: `"lstm"`)
- `activation`: Activation function - `"relu"`, `"tanh"`, or `"sigmoid"` (default: `"relu"`)
- `optimizer_name`: Optimizer - `"adam"`, `"sgd"`, or `"rmsprop"` (default: `"adam"`)
- `use_grad_clip`: Enable gradient clipping (default: `False`)

**Example with custom settings:**

Edit the `main()` function in `src/train.py` to set your desired hyperparameters, then run:

```bash
python -m src.train
```

### Evaluation

To evaluate a trained model on the test set:

```bash
python -m src.evaluate
```

**Configuration options** (edit in `src/evaluate.py` lines ~33-42):

- `data_dir`: Path to the IMDb dataset (default: `"data/aclImdb"`)
- `seq_len`: Sequence length (must match training; default: 50)
- `batch_size`: Batch size for evaluation (default: 32)
- `model_path`: Path to the trained model file (default: `"results/sentiment_model.pth"`)

This script will load a trained model and output:
- **Accuracy** on the test set
- **F1-score** on the test set

## Expected Output Files

After training, the following files are generated in the `results/` directory:

### Model Files
- `model_<rnn_type>_seq<length>.pth` - Trained model weights
  - Example: `model_lstm_seq50.pth`

### Loss & Metrics
- `loss_<rnn_type>_seq<length>.csv` - Per-epoch training and validation losses
  - Example: `loss_lstm_seq50.csv`
- `metrics.csv` - Summary row with test accuracy, F1-score, and runtime

### Visualizations
- `plots/loss_<rnn_type>_seq<length>.png` - Loss curve plot (training and validation)
  - Example: `plots/loss_lstm_seq50.png`

### Example Output Structure
```
results/
├── model_lstm_seq50.pth
├── loss_lstm_seq50.csv
├── metrics.csv
└── plots/
    └── loss_lstm_seq50.png
```

## Expected Runtime

Training times vary based on hardware and configuration:

| Model | Seq Length | Epochs | Approx. Time |
|-------|-----------|--------|-------------|
| RNN | 50 | 5 | ~2-3 min |
| LSTM | 50 | 5 | ~3-4 min |
| BiLSTM | 50 | 5 | ~5-7 min |
| LSTM | 25 | 5 | ~1-2 min |
| LSTM | 100 | 5 | ~4-6 min |

**Note**: Times are approximate for CPU-only training on a standard machine. GPU acceleration is not implemented in this version.

Each epoch includes:
- Training on ~19,500 samples (with validation holdout)
- Validation on ~5,500 samples
- Progress updates every 100 batches

## Project Structure

```
rnn-sentiment/
├── src/
│   ├── train.py           # Main training script
│   ├── evaluate.py        # Evaluation script
│   ├── models.py          # RNN/LSTM/BiLSTM model definitions
│   ├── preprocess.py      # Data loading and preprocessing
│   ├── utils.py           # Utility functions
│   ├── make_plot.py       # Plotting utilities
│   └── requirements.txt   # Dependencies
├── data/
│   └── aclImdb/          # IMDb dataset
├── results/              # Output directory (created during training)
├── README.md             # This file
└── report.md             # Experimental results and analysis
```

## Model Architecture

All models use the following architecture:

- **Embedding Layer**: Converts word IDs to 100-dimensional vectors
- **RNN/LSTM/BiLSTM Layer**: 2 stacked recurrent layers with 64 hidden units
- **Dropout**: 0.5 dropout for regularization
- **Fully Connected Layer**: Maps RNN output to binary classification
- **Output**: Sigmoid activation for probability [0, 1]
- **Loss**: Binary Cross-Entropy (BCE)

### Supported Architectures

- **RNN**: Basic Recurrent Neural Network
- **LSTM**: Long Short-Term Memory (solves vanishing gradient problem)
- **BiLSTM**: Bidirectional LSTM (processes sequence in both directions)

## Preprocessing Details

1. **Text Cleaning**: Lowercase, remove punctuation, normalize spaces
2. **Tokenization**: NLTK word tokenizer
3. **Vocabulary**: Top 10,000 most frequent words
4. **Sequences**: Padded/truncated to fixed length (25, 50, or 100 tokens)
5. **Train/Val Split**: 90% training, 10% validation

## Reproducibility

The scripts use fixed random seeds (default: 42) for reproducibility:

```python
set_seed(42)  # in train.py
```

This ensures consistent results across multiple runs.

