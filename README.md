# SimpleRNN — IMDB Sentiment Classification

Lightweight repository demonstrating an RNN-based approach for sentiment classification (IMDB). Contains training and inference notebooks, a small demo script, and a pre-trained Keras model.

## Repository contents
- `simplernn.ipynb` — primary training notebook (model architecture, training loop, evaluation).
- `embedding.ipynb` — experiments using embedding layers / pre-trained embeddings.
- `prediction.ipynb` — inference examples showing how to preprocess text and obtain predictions.
- `main.py` — quick demo script (load model and run sample inference).
- `simple_rnn_imdb.h5` — pre-trained Keras model (HDF5).
- `requirements.txt` — Python package requirements.
- `LICENSE` — project license.

## Quickstart

1. Create and activate virtual environment (Windows):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt