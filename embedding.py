from __future__ import annotations
from gensim.models import Word2Vec
from typing import List

def train_word2vec(sentences: List[List[str]], vector_size=200, window=5, min_count=2,
                   workers=4, sg=1, epochs=10) -> Word2Vec:
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
    )
    return model

def save_model(model: Word2Vec, out_path: str) -> None:
    model.save(out_path)
