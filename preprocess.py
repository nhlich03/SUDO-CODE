from __future__ import annotations
import re
from typing import List
try:
    from underthesea import word_tokenize, lemmatize
except Exception:  # pragma: no cover
    word_tokenize = None
    lemmatize = None

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"(https?://\S+|www\.\S+)", " ", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_vi(text: str) -> List[str]:
    if word_tokenize is None:
        return text.split()
    return word_tokenize(text, format="text").split()

def optional_lemmatize(tokens: List[str]) -> List[str]:
    if lemmatize is None:
        return tokens
    pairs = lemmatize(" ".join(tokens))
    lemmas = []
    for pair in pairs:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            lemmas.append(pair[1])
        else:
            lemmas.append(str(pair))
    return lemmas

def preprocess_corpus(texts, use_lemmatize: bool = True) -> list[list[str]]:
    processed = []
    for t in texts:
        t2 = clean_text(t)
        toks = tokenize_vi(t2)
        if use_lemmatize:
            toks = optional_lemmatize(toks)
        processed.append(toks)
    return processed
