import os
import re
import json
from collections import Counter
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Optional parallel pre-processing
try:
    from pandarallel import pandarallel
    from multiprocessing import cpu_count
    pandarallel.initialize(progress_bar=True, nb_workers=cpu_count())
    _PANDARALLEL = True
except Exception:
    _PANDARALLEL = False

# ---------------- Preprocessing ----------------
class Preprocessing:
    def __init__(self, file_path):
        self.stopwords = self.read_stopwords(file_path)

        self.re_k     = re.compile(r'\b(\d+)k\b')
        self.re_pcent = re.compile(r'\b(\d+)%\b')
        self.re_m     = re.compile(r'\b(\d+)m\b')
        self.re_s     = re.compile(r'\b(\d+)s\b')
        self.re_min   = re.compile(r"\b(\d+)'\b")
        self.re_h     = re.compile(r'\b(\d+)h\b')
        self.re_non   = re.compile(r'[^0-9a-zA-ZÀ-ỹ\s]')

    def read_stopwords(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())

    def preprocessing_text(self, text: str) -> str:
        t = str(text).lower().strip()
        t = self.re_k.sub(r'\1 ngàn', t)
        t = self.re_pcent.sub(r'\1 phần trăm', t)
        t = self.re_m.sub(r'\1 mét', t)
        t = self.re_s.sub(r'\1 giây', t)
        t = self.re_min.sub(r'\1 phút', t)
        t = self.re_h.sub(r'\1 giờ', t)
        t = self.re_non.sub(' ', t)
        tokens = [w for w in t.split() if w not in self.stopwords]
        return " ".join(tokens)

# ---------------- Vocab ----------------
class Vocab:
    def __init__(self, max_size=50000, min_freq=1, pad_token="<PAD>", oov_token="<OOV>"):
        self.max_size = max_size
        self.min_freq = min_freq
        self.pad_token = pad_token
        self.oov_token = oov_token
        self.pad_idx, self.oov_idx = 0, 1
        self.itos = []
        self.stoi = {}

    def build(self, texts: List[str]):
        counter = Counter()
        for s in texts:
            counter.update(self.tokenize(s))

        most_common = counter.most_common()
        if self.min_freq > 1:
            most_common = [kv for kv in most_common if kv[1] >= self.min_freq]
        most_common = most_common[: self.max_size]

        self.itos = [self.pad_token, self.oov_token] + [w for w, _ in most_common]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        return self

    def tokenize(self, s: str):
        return s.strip().split()

    def numericalize(self, text: str):
        return [self.stoi.get(tok, self.oov_idx) for tok in self.tokenize(text)]

    def denumericalize(self, ids):
        return [self.itos[i] if i < len(self.itos) else self.oov_token for i in ids]

    def save(self, path="outputs/vocab.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False)

    def load(self, path="outputs/vocab.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.itos = data["itos"]
            self.stoi = {w: i for i, w in enumerate(self.itos)}
        return self

# ---------------- Dataset & Collate ----------------
class TxtClsDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocab, pad_idx=0):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.pad_idx = pad_idx
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        ids = torch.tensor(self.vocab.numericalize(self.texts[idx]), dtype=torch.long)
        y   = torch.tensor(self.labels[idx], dtype=torch.long)
        return ids, y

def collate_fn(batch, pad_idx=0):
    seqs, ys = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True, padding_value=pad_idx)
    ys = torch.stack(ys)
    return padded, lengths, ys

# ---------------- Loader builder ----------------
def build_loaders_and_assets(cfg, logger):
    import pandas as pd

    train_path = cfg["data"]["train_path"]
    test_path  = cfg["data"]["test_path"]
    text_col   = cfg["data"]["text_col"]
    raw_col    = cfg["data"]["raw_text_col"]
    label_col  = cfg["data"]["label_col"]
    stopwords_path = cfg["data"]["stopwords_path"]

    # read CSV
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # preprocess to text_col (like notebook)
    pre = Preprocessing(stopwords_path)
    if _PANDARALLEL:
        train_df[text_col] = train_df[raw_col].astype(str).parallel_apply(pre.preprocessing_text)
        test_df[text_col]  = test_df[raw_col].astype(str).parallel_apply(pre.preprocessing_text)
    else:
        train_df[text_col] = train_df[raw_col].astype(str).apply(pre.preprocessing_text)
        test_df[text_col]  = test_df[raw_col].astype(str).apply(pre.preprocessing_text)

    # split
    val_size = cfg["data"]["val_size"]
    random_state = cfg["data"]["random_state"]
    stratify = train_df[label_col] if cfg["data"]["stratify"] else None
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state, stratify=stratify)

    # dropna
    train_df = train_df.dropna(subset=[text_col, label_col])
    val_df   = val_df.dropna(subset=[text_col, label_col])
    test_df  = test_df.dropna(subset=[text_col, label_col])

    X_train, y_train_raw = train_df[text_col].astype(str).tolist(), train_df[label_col].astype(str).tolist()
    X_val,   y_val_raw   = val_df[text_col].astype(str).tolist(),   val_df[label_col].astype(str).tolist()
    X_test,  y_test_raw  = test_df[text_col].astype(str).tolist(),  test_df[label_col].astype(str).tolist()

    # label encoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val   = le.transform(y_val_raw)
    y_test  = le.transform(y_test_raw)
    num_classes = len(le.classes_)
    logger.info(f"Classes: {list(le.classes_)}")

    # vocab
    vocab = Vocab(max_size=cfg["data"]["max_vocab"], min_freq=cfg["data"]["min_freq"]).build(X_train)
    os.makedirs("outputs", exist_ok=True)
    vocab.save("outputs/vocab.json")
    with open("outputs/label_classes.json","w",encoding="utf-8") as f:
        json.dump({"classes": list(le.classes_)}, f, ensure_ascii=False)

    PAD = cfg["data"]["PAD"]

    # datasets & loaders
    train_ds = TxtClsDataset(X_train, y_train, vocab, pad_idx=PAD)
    val_ds   = TxtClsDataset(X_val,   y_val,   vocab, pad_idx=PAD)
    test_ds  = TxtClsDataset(X_test,  y_test,  vocab, pad_idx=PAD)

    batch_size  = cfg["loader"]["batch_size"]
    num_workers = cfg["loader"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, collate_fn=lambda b: collate_fn(b, pad_idx=PAD))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda b: collate_fn(b, pad_idx=PAD))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda b: collate_fn(b, pad_idx=PAD))

    assets = {"vocab": vocab, "label_encoder": le, "num_classes": num_classes, "PAD": PAD}
    return train_loader, val_loader, test_loader, assets
