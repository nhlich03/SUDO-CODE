
from typing import Tuple, Literal, Optional
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ModelName = Literal["svm", "nb"]

def build_pipeline(
    model: ModelName = "svm",
    ngram_min: int = 1,
    ngram_max: int = 2,
    min_df: int = 2,
    max_df: float = 0.9
) -> Pipeline:
    vec = TfidfVectorizer(ngram_range=(ngram_min, ngram_max), min_df=min_df, max_df=max_df)
    if model == "nb":
        clf = MultinomialNB()
    elif model == "svm":
        clf = LinearSVC()
    else:
        raise ValueError(f"Unsupported model: {model}")
    return Pipeline([("tfidf", vec), ("clf", clf)])
