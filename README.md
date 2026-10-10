# SentimentScope: Transformer Architecture for Movie Review Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Dataset](https://img.shields.io/badge/Dataset-Stanford%20IMDB-blue.svg)](https://ai.stanford.edu/~amaas/data/sentiment/)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-77.08%25-brightgreen.svg)]()

A deep learning project implementing a custom Transformer decoder architecture from scratch in **PyTorch** and adapting it for sequence classification. Fine-tuned on the **Stanford Large Movie Review Dataset (IMDB)** to classify sentiment polarity with high accuracy and low inference latency.

Developed as part of the **Udacity AI / Deep Learning Nanodegree**.

---

## 📌 Project Overview

Traditional sequence models often struggle with long-range dependencies and bidirectional contextual understanding. In this project, a modular Transformer model (`DemoGPT`) is engineered from fundamental building blocks (multi-head attention, layer normalization, residual feed-forward networks) and adapted for binary sentiment classification.

### Key Objectives:
- **Architecture from Scratch:** Implement scaled dot-product attention, multi-head attention mechanisms, and stacked transformer blocks without high-level black-box abstractions.
- **Classification Adaptation:** Adapt generative decoder layers for classification using sequence mean-pooling and linear projection heads.
- **End-to-End Pipeline:** Construct custom PyTorch `Dataset` and `DataLoader` pipelines with subword tokenization (`bert-base-uncased`), batch shuffling, and dynamic padding/truncation.
- **Evaluation & Benchmarking:** Attain over 75% accuracy on 25,000 unseen test reviews under strict parameter constraints.

---

## 🏗️ Architecture & Technical Specifications

```
                     [ Raw Input Review ]
                              │
                              ▼
           [ AutoTokenizer ('bert-base-uncased') ]
                              │
                              ▼
           [ Token IDs + Learned Positional Embeddings ]
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │       Stacked Transformer Blocks       │
         │  ┌──────────────────────────────────┐  │
         │  │ Pre-LN + Multi-Head Attention    │  │ x 4 Layers
         │  │ Residual Addition                │  │
         │  │ Pre-LN + GELU Feed-Forward       │  │
         │  │ Residual Addition                │  │
         │  └──────────────────────────────────┘  │
         └────────────────────────────────────────┘
                              │
                              ▼
                    [ Layer Normalization ]
                              │
                              ▼
           [ Mean Pooling across Sequence Dimension ]
                              │
                              ▼
             [ Linear Classifier (d_embed -> 2) ]
                              │
                              ▼
                    [ Logits: Pos / Neg ]
```

### Model Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Embedding Dimension (`d_embed`)** | `128` | Latent representation dimension |
| **Context Length (`context_size`)** | `128` | Maximum sequence length (tokens) |
| **Transformer Layers (`layers_num`)** | `4` | Number of stacked Transformer blocks |
| **Attention Heads (`heads_num`)** | `4` | Parallel self-attention heads |
| **Head Dimension (`head_size`)** | `32` | Head dimension ($heads \times head\_size = 128$) |
| **Vocabulary Size** | `30,522` | Tokenizer vocabulary (`bert-base-uncased`) |
| **Activation Function** | `GELU` | Gaussian Error Linear Unit in FFN |
| **Dropout Rate** | `0.15` | Regularization across attention & FFN |
| **Output Classes** | `2` | Binary classification (0: Negative, 1: Positive) |

---

## 📊 Dataset & Preprocessing

- **Dataset:** Stanford Large Movie Review Dataset (IMDB)
- **Total Samples:** 50,000 reviews evenly split (25,000 Train / 25,000 Test)
- **Data Splits:**
  - **Train:** 22,500 samples
  - **Validation:** 2,500 samples
  - **Test:** 25,000 samples
- **Tokenization:** Subword WordPiece tokenization via `AutoTokenizer.from_pretrained('bert-base-uncased')`
- **Data Loading:** Custom PyTorch `IMDBDataset` with `torch.utils.data.DataLoader` (Batch Size = 32, shuffled per epoch)

---

## 🚀 Training & Experimental Results

- **Optimizer:** `AdamW` (learning rate = `3e-4`)
- **Loss Function:** `nn.CrossEntropyLoss()`
- **Hardware:** CUDA-enabled GPU acceleration
- **Epochs:** 5

### Validation Progress

| Epoch | Step Loss (Avg) | Validation Accuracy |
| :---: | :---: | :---: |
| **Untrained Baseline** | — | `50.32%` (Random Guessing) |
| **Epoch 1** | `0.5581` | `69.12%` |
| **Epoch 2** | `0.4787` | `76.32%` |
| **Epoch 3** | `0.4216` | `77.08%` |
| **Epoch 4** | `0.3711` | `78.04%` |
| **Epoch 5** | `0.3286` | **`78.68%`** |

### Final Evaluation on Unseen Test Set

```text
============================================================
Test Set Size: 25,000 reviews
Final Test Accuracy: 77.08%
Target Benchmark:    > 75.00% [PASSED]
Checkpoint Saved:    sentiment_model_checkpoint.pt
============================================================
```

---

## 📁 Repository Structure

```
.
├── SentimentScope_starter.ipynb      # Complete notebook containing architecture, training & evaluation
├── sentiment_model_checkpoint.pt     # Trained PyTorch model weights (evaluated at 77.08% accuracy)
├── .gitignore                        # Git ignore rules for archives & environments
└── README.md                         # Project documentation and performance metrics
```

---

## 🛠️ Quickstart & Reproduction

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis
```

### 2. Environment Setup
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install torch torchvision transformers pandas numpy matplotlib
```

### 3. Dataset Download & Extraction
Download the dataset archive into the root directory:
```bash
tar -xzf aclImdb_v1.tar.gz
```

### 4. Run Inference or Retrain
Open the notebook in Jupyter or VS Code:
```bash
jupyter notebook SentimentScope_starter.ipynb
```

To load the trained model in PyTorch:
```python
import torch
from SentimentScope_starter import DemoGPT, config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DemoGPT(config).to(device)
model.load_state_dict(torch.load("sentiment_model_checkpoint.pt", map_location=device))
model.eval()
```

---

## 💡 Key Engineering Takeaways

1. **Decoder-to-Classifier Adaptation:** Unlike autoregressive language models that decode tokens sequentially, sequence classifiers aggregate the hidden representations. Applying mean pooling across sequence tokens provided smooth, gradient-stable representations compared to single-token extraction without pre-training.
2. **Subword Tokenization Advantages:** Subword tokenization balances compact vocabulary footprint with zero out-of-vocabulary (OOV) errors, capturing nuanced movie terminology without sparse word vectors.
3. **Training Stability:** Combining residual connections, pre-layer normalization (`LayerNorm`), and `AdamW` weight decay prevented gradient vanishing across 4 stacked attention layers.

---

## 📜 License & Acknowledgments
- Project created as part of the **Udacity Machine Learning / AI Nanodegree**.
- Dataset provided by Stanford AI Lab (Andrew Maas et al.).
