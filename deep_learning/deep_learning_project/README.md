# deep-learning (Text Classification, PyTorch)

This is your notebook split into `.py` files for **Vietnamese text classification** with a simple average-embedding MLP head.

## Structure
```
deep-learning/
  main.py
  data.py
  model.py
  train.py
  eval.py
  utils.py
  configs/
    config.yaml
  outputs/
    checkpoints/
    logs/
    reports/
  requirements.txt
  README.md
```

## What matches your notebook
- Preprocessing (regex replacements + stopwords) in `data.py::Preprocessing`
- Vocabulary building with `Vocab` (PAD=0, OOV=1) and `numericalize`
- `TxtClsDataset` + `collate_fn` returning `(padded_ids, lengths, labels)`
- `TextClassifier` with averaged embeddings → FC → ReLU → Dropout → FC
- `Trainer` loop with early stopping on val accuracy and saving `best.pth`
- Classification report and confusion matrix in `outputs/reports/` after eval
- LabelEncoder classes saved to `outputs/label_classes.json`

## Run
1) Install deps
```
pip install -r requirements.txt
```

2) Edit paths in `configs/config.yaml` to point to your CSV & stopwords.

3) Train
```
python main.py --config configs/config.yaml train
```

4) Evaluate
```
python main.py --config configs/config.yaml eval
```

5) (Optional) CSV inference
```
python main.py --config configs/config.yaml infer --input /path/to/new.csv
```
