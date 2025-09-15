import os
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import json
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# Load raw books dataset
def load_books(data_directory: str) -> pd.DataFrame:
    texts, titles, authors = [], [], []
    files = sorted(os.listdir(data_directory))
    for file_name in files:
        file_path = os.path.join(data_directory, file_name)
        if not file_path.endswith(".txt"):
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        texts.append(content)
        parts = file_name.replace(".txt", "").split("-")
        title = "-".join(parts[:-1]) if len(parts) > 1 else parts[0]
        author = parts[-1] if len(parts) > 1 else ""
        titles.append(title)
        authors.append(author)
    return pd.DataFrame({"Titles": titles, "Texts": texts, "Authors": authors})


class DataSplitter:
    def __init__(self, train_ratio=0.7, val_ratio=0.1, seed=42, shuffle=True):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.shuffle = shuffle

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = data.copy()
        if self.shuffle:
            df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        n = len(df)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        train = df.iloc[:n_train]
        val = df.iloc[n_train : n_train + n_val]
        test = df.iloc[n_train + n_val :]
        return train, val, test
    

# Preprocessing
class TextPreprocessor:
    QUOTE_MAP = {
        "“": '"', "”": '"', "„": '"', "‟": '"', "〝": '"', "〞": '"',
        "‘": "'", "’": "'", "‚": "'", "‛": "'", "´": "'", "`": "'",
    }

    def __init__(self, para_token="<PARA>", bos_token="<BOS>", eos_token="<EOS>", min_words=200, drop_patterns: Optional[List[str]] = None):
        self.PARA, self.BOS, self.EOS = para_token, bos_token, eos_token
        self.min_words = min_words
        self.drop_res = [re.compile(p) for p in (drop_patterns or [])]
        self.re_ctrl = re.compile(r"[\u0000-\u0008\u000B-\u000C\u000E-\u001F]")
        self.re_spaces = re.compile(r"[ \t]+")
        self.re_manynl = re.compile(r"\n{3,}")
        self.re_ws_nl = re.compile(r"[ \t]+(\n)|(\n)[ \t]+")
        self.re_space_before_punct = re.compile(r"\s+([.,!?;:])")
        self.re_need_space_after = re.compile(r"([.,!?;:])([^\s])")

    def _normalize_unicode(self, s: str) -> str:
        s = unicodedata.normalize("NFC", s or "")
        for k, v in self.QUOTE_MAP.items():
            s = s.replace(k, v)
        return s

    def _strip_boilerplate(self, s: str) -> str:
        if not self.drop_res:
            return s
        keep_lines = []
        for line in s.splitlines():
            if any(rgx.search(line) for rgx in self.drop_res):
                continue
            keep_lines.append(line)
        return "\n".join(keep_lines)

    def clean_text(self, s: str) -> str:
        s = self._normalize_unicode(s)
        s = self._strip_boilerplate(s)
        s = self.re_ctrl.sub(" ", s)
        s = self.re_spaces.sub(" ", s)
        s = self.re_manynl.sub("\n\n", s)
        s = self.re_ws_nl.sub(lambda m: "\n", s)
        s = s.strip()
        s = s.replace("\n\n", f" {self.PARA} ").replace("\n", " ")
        s = self.re_space_before_punct.sub(r"\1", s)
        s = self.re_need_space_after.sub(r"\1 \2", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def process_series(self, texts: Iterable[str]) -> List[str]:
        records: List[str] = []
        for t in texts:
            clean = self.clean_text(str(t))
            if len(clean.split()) >= self.min_words:
                records.append(f"{self.BOS} {clean} {self.EOS}")
        return records


# Tokenizer & Vocab
class Tokenizer:
    def __init__(self, specials: List[str] = ["<BOS>", "<EOS>", "<PARA>"]):
        esc = [re.escape(s) for s in specials]
        specials_pat = "|".join(esc)
        self.tok_re = re.compile(rf"({specials_pat}|\.\.\.|…|[.,!?;:()\[\]\"'“”‘’—-]|[\w\-]+)", flags=re.UNICODE)

    def tokenize(self, s: str) -> List[str]:
        s = unicodedata.normalize("NFC", s or "")
        return [t for t in self.tok_re.findall(s) if t.strip()]


PAD, UNK, BOS, EOS, PARA = "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<PARA>"


class Vocab:
    def __init__(self, min_freq=2, max_size=60000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

    def build(self, texts: Iterable[str], tokenizer: Tokenizer):
        counter = Counter()
        for s in texts:
            counter.update(tokenizer.tokenize(s))
        specials = [PAD, UNK, BOS, EOS, PARA]
        for sp in specials:
            counter[sp] += 10**9
        vocab = [w for w, c in counter.items() if c >= (0 if w in specials else self.min_freq)]
        vocab.sort(key=lambda w: (-counter[w], w))
        if self.max_size:
            vocab = vocab[: self.max_size]
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.stoi.get(UNK)
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos.get(i, UNK) for i in ids]

    def save(self, path: str):
        obj = {"stoi": self.stoi, "itos": self.itos}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        v = Vocab()
        v.stoi = {k: int(v_) if isinstance(v_, str) and v_.isdigit() else v_ for k, v_ in obj["stoi"].items()}
        v.itos = {int(k): v_ for k, v_ in obj["itos"].items()}
        return v


# Dataset + DataLoader

class LMDataset(Dataset):
    def __init__(self, records: List[str], tokenizer: Tokenizer, vocab: Vocab, max_len: int = 256, stride: Optional[int] = None, drop_last_short: bool = True):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride or max_len
        self.samples: List[List[int]] = []
        for s in records:
            ids = vocab.encode(tokenizer.tokenize(s))
            for i in range(0, max(0, len(ids) - 1), self.stride):
                chunk = ids[i : i + max_len]
                if drop_last_short and len(chunk) < 4:
                    continue
                if len(chunk) >= 2:
                    self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.samples[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


def lm_collate(batch, pad_id: int):
    seqs, ys = zip(*batch)
    padded_x = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    padded_y = pad_sequence(ys, batch_first=True, padding_value=pad_id)
    mask = (padded_x != pad_id).long()
    return padded_x, padded_y, mask


def make_loaders(records_split: Tuple[List[str], List[str], List[str]], tokenizer: Tokenizer, vocab: Vocab, max_len: int, stride: int, batch_size: int, pad_id: int):
    train_rec, val_rec, test_rec = records_split
    train_ds = LMDataset(train_rec, tokenizer, vocab, max_len=max_len, stride=stride)
    val_ds = LMDataset(val_rec, tokenizer, vocab, max_len=max_len, stride=stride)
    test_ds = LMDataset(test_rec, tokenizer, vocab, max_len=max_len, stride=stride)
    collate = lambda b: lm_collate(b, pad_id=pad_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, test_loader