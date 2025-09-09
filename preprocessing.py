import re
from typing import List, Iterable
from underthesea import text_normalize as vn_normalize, word_tokenize

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"(https?://\S+|www\.\S+)", "", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def optional_lemmatize(tokens: List[str]) -> List[str]:
    try:
        from underthesea import lemmatize
        lemmas = []
        for pair in lemmatize(" ".join(tokens)):
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                lemmas.append(pair[1])
            else:
                lemmas.append(str(pair))
        return lemmas
    except Exception:
        return tokens

def tokenize_vi(text: str, stopwords: Iterable[str]) -> List[str]:
    if not text:
        return []
    text = vn_normalize(text)
    text = clean_text(text)
    toks = word_tokenize(text, format="text").split()
    toks = [t for t in toks if t not in stopwords and not t.isdigit()]
    toks = optional_lemmatize(toks)
    return toks

def process_text(text: str, stopwords: Iterable[str]) -> str:
    return " ".join(tokenize_vi(text, stopwords))
