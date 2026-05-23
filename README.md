# Spam Email Classification with CNN

Classifies emails as spam or ham (not spam) using a 1D Convolutional Neural Network operating on text embeddings. Achieves high accuracy by learning local n-gram patterns in email content.

## Overview

Unlike bag-of-words approaches, CNNs applied to text capture local sequential patterns (phrases, word combinations) that are indicative of spam. This project tokenises email text, embeds it into dense vectors, and applies 1D convolutions to extract discriminative features.

## Tech Stack

- **Language:** Python 3
- **Libraries:** TensorFlow/Keras, NumPy, JSON, scikit-learn, matplotlib
- **Dataset:** `full_spam_dataset.json`
- **Model format:** `.keras`
- **Environment:** Jupyter Notebook

## Key Concepts

- Text tokenisation and sequence padding
- Embedding layer (learned word vectors)
- 1D convolutional filters for text feature extraction
- Global max pooling
- Binary cross-entropy loss, sigmoid output
- Evaluation: accuracy, precision, recall, AUC-ROC

## Project Structure

```
ai-spam-email-classification-cnn/
├── spam_email_classification_cnn.ipynb   # Main notebook
├── full_spam_dataset.json                          # Email dataset (spam + ham)
└── spam_cnn_model.keras                             # Trained CNN model
```

## How to Run

```bash
pip install tensorflow numpy scikit-learn matplotlib jupyter
jupyter notebook spam_email_classification_cnn.ipynb
```

Load the saved model for inference:

```python
from tensorflow import keras
model = keras.models.load_model("spam_cnn_model.keras")
```

## Model Architecture

```
Input (padded sequences)
  → Embedding (vocab_size × embed_dim)
  → Conv1D (filters, kernel_size) + ReLU
  → GlobalMaxPooling1D
  → Dense (hidden) + ReLU + Dropout
  → Dense (1) + Sigmoid
```
