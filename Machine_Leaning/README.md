
# Sentiment Classifier (SVM / Naive Bayes)

This project performs **sentiment classification** (positive / negative) from feedback using 2 models:
- **SVM** (LinearSVC)  
- **Naive Bayes** (MultinomialNB)  

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configure
Edit `config.yaml`:
```yaml
train_path: data/UIT/train.csv
test_path: data/UIT/test.csv
text_col: comment
label_col: label
model: svm   # or nb
tfidf:
  ngram_min: 1
  ngram_max: 2
  min_df: 2
  max_df: 0.9
artifacts_dir: artifacts
outputs_dir: outputs
model_name: null
slang_path: replace_list.json
```

replace_list is in `replace_list.json`.

## Run
Using YAML only:
```bash
python -m src.main --config config.yaml
```

Override model or params on the fly:
```bash
python -m src.main --config config.yaml --model nb --ngram_max 3
```

Outputs:
- Saved model: `artifacts/model_<name>.pkl`
- Predictions: `outputs/predictions_<name>.csv`
- Metrics: `outputs/metrics_<name>.csv` and `outputs/metrics_<name>.json`
```
