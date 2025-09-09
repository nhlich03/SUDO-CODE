from __future__ import annotations
from typing import Optional
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_embeddings(model, n_samples: int = 500, perplexity: float = 30.0,
                    random_state: int = 42, output_path: Optional[str] = None):
    words = list(model.wv.key_to_index.keys())
    if not words:
        raise ValueError("Model has no vocabulary.")

    sample = words[:]
    random.Random(random_state).shuffle(sample)
    sample = sample[:n_samples]

    X = np.asarray([model.wv[w] for w in sample], dtype=np.float32)

    max_valid_perp = max(1.0, min(float(perplexity), float(len(sample) - 1)))
    if max_valid_perp != perplexity:
        perplexity = max_valid_perp 

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="random")
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=10)
    for i, w in enumerate(sample):
        plt.annotate(w, (X_2d[i, 0], X_2d[i, 1]), fontsize=6)

    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=200)
    else:
        plt.show()
