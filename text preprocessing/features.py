import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def build_vectorizers(min_df=2, max_df=0.9, ngrams_uni=(1,1), ngrams_bi=(1,2), tfidf_norm="l2"):
    cv_uni = CountVectorizer(ngram_range=ngrams_uni, min_df=min_df, max_df=max_df)
    cv_bi  = CountVectorizer(ngram_range=ngrams_bi,  min_df=min_df, max_df=max_df)
    tf_uni = TfidfVectorizer(ngram_range=ngrams_uni, min_df=min_df, max_df=max_df, norm=tfidf_norm)
    tf_bi  = TfidfVectorizer(ngram_range=ngrams_bi,  min_df=min_df, max_df=max_df, norm=tfidf_norm)
    return cv_uni, cv_bi, tf_uni, tf_bi

def corpus_stats(proc_series: pd.Series) -> pd.DataFrame:
    tok_counts = proc_series.str.split().apply(len)
    return pd.DataFrame({
        "num_docs":[len(proc_series)],
        "avg_tokens":[tok_counts.mean()],
        "median_tokens":[tok_counts.median()],
        "min_tokens":[tok_counts.min()],
        "max_tokens":[tok_counts.max()],
    })

def matrix_report(X, name: str) -> pd.Series:
    nnz = X.nnz
    shape = X.shape
    sparsity = 1 - nnz / (shape[0] * shape[1])
    return pd.Series({
        "representation": name,
        "num_docs": shape[0],
        "vocab_size": shape[1],
        "nonzeros": nnz,
        "sparsity": round(float(sparsity), 4)
    })

def top_terms_count(cv, X, k=20) -> pd.DataFrame:
    vocab = np.array(cv.get_feature_names_out())
    freqs = np.asarray(X.sum(axis=0)).ravel()
    idx = freqs.argsort()[::-1][:k]
    return pd.DataFrame({"term": vocab[idx], "freq": freqs[idx].astype(int)})

def top_terms_tfidf(tfv, X, k=20) -> pd.DataFrame:
    vocab = np.array(tfv.get_feature_names_out())
    max_w = X.max(axis=0).toarray().ravel()
    idx = max_w.argsort()[::-1][:k]
    return pd.DataFrame({"term": vocab[idx], "max_tfidf": np.round(max_w[idx], 4)})
