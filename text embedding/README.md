# Vietnamese Word2Vec Embedding Project

A project to train and visualize Word2Vec embeddings on Vietnamese text.

---

## Overview

The workflow includes:

1. **Text preprocessing**:
   - Cleaning (remove URLs, emails, special characters).
   - Vietnamese tokenization (via `underthesea`).
   - Optional lemmatization.
2. **Word2Vec Training**:
   - Skip-gram (sg=1) or CBOW (sg=0).
   - Configurable vector size, window, epochs, etc.
3. **Visualization**:
   - 2D projection of embeddings using t-SNE.
   - Save to PNG or show inline.
4. **Saving results**:
   - Trained Word2Vec model (`.model`).
   - Optional embedding visualization in `reports/figures/`.

---

## Project Structure

```
.
├─ config.yaml        # Configurable parameters (data, preprocess, training, visualization)
├─ main.py            # Entry point to run the pipeline
├─ dataset.py         # Data loading
├─ preprocess.py      # Cleaning, tokenization, lemmatization
├─ embedding.py       # Word2Vec training & saving
├─ visualize.py       # t-SNE plotting for embeddings
├─ models/            # Saved Word2Vec models
├─ data.txt           # Txt file
├─ reports/
│  └─ figures/        # Visualization outputs (png)
├─ requirements.txt   # Dependencies
└─ README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run training
```bash
python main.py
```

This will:
- Load data and preprocess.
- Train a Word2Vec model.
- Save model to `models/w2v.model`.
- Generate visualization (if enabled in config).

---

## Configuration

All settings are in `config.yaml`:

```yaml
data:
  input_path: "data/raw/sample.txt"   # Input file (.txt or .csv)
  text_column: "text"                 # Only used if CSV

preprocess:
  use_lemmatize: true                 # Enable/disable lemmatization

train:
  vector_size: 200
  window: 5
  min_count: 2
  workers: 4
  sg: 1                               # 1 = skip-gram, 0 = CBOW
  epochs: 10

visualize:
  n_samples: 500
  perplexity: 30
  random_state: 42
  output_path: "reports/figures/tsne.png"
```

---

## Outputs

- **Model**: `models/w2v.model`
- **Visualization**: `reports/figures/tsne.png` (if configured)

---

## Notes

- For small datasets, reduce `perplexity` in `config.yaml` (must be `< number of words`).
- Ensure `underthesea` is installed for Vietnamese tokenization.
