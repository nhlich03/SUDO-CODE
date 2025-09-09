import yaml
import pandas as pd

from utils import (
    download_dataset_kaggle, load_json_as_df, load_stopwords_txt,
    output_results
)
from preprocessing import process_text
from features import (
    build_vectorizers, corpus_stats, matrix_report,
    top_terms_count, top_terms_tfidf
)

# Load config.yaml
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_pipeline(save_flag: bool = False, print_flag: bool = True):
    config = load_config()

    # Dataset
    ds_dir = download_dataset_kaggle()
    df = load_json_as_df(ds_dir, config["dataset"]["file"])

    # Clean
    df = df.dropna().drop_duplicates()

    # Sample
    df_sample = df.sample(
        n=config["sample"]["size"],
        random_state=config["sample"]["random_state"]
    )

    # Stopwords
    vn_stop = load_stopwords_txt(config["stopwords_txt"])

    # Preprocess
    df_sample["processed_text"] = df_sample["content"].apply(lambda x: process_text(x, vn_stop))
    processed = df_sample["processed_text"]

    # Vectorizers
    cv_uni, cv_bi, tf_uni, tf_bi = build_vectorizers(
        min_df=config["vectorizer"]["min_df"],
        max_df=config["vectorizer"]["max_df"],
        ngrams_uni=tuple(config["vectorizer"]["ngrams_uni"]),
        ngrams_bi=tuple(config["vectorizer"]["ngrams_bi"]),
        tfidf_norm=config["vectorizer"]["tfidf_norm"]
    )
    X_bow_uni   = cv_uni.fit_transform(processed)
    X_bow_bi    = cv_bi.fit_transform(processed)
    X_tfidf_uni = tf_uni.fit_transform(processed)
    X_tfidf_bi  = tf_bi.fit_transform(processed)

    # Results
    output_results(processed.to_frame(), "Processed Text", "processed_text.csv", save_flag, print_flag)

    cs_df = corpus_stats(processed).round(2)
    output_results(cs_df, "Corpus stats", "corpus_stats.csv", save_flag, print_flag)

    rep_df = pd.concat([
        matrix_report(X_bow_uni,  "BoW unigram"),
        matrix_report(X_bow_bi,   "BoW uni+bi"),
        matrix_report(X_tfidf_uni,"TF-IDF unigram"),
        matrix_report(X_tfidf_bi, "TF-IDF uni+bi"),
    ], axis=1).T
    output_results(rep_df, "Representations", "representations.csv", save_flag, print_flag)

    top_bow_uni = top_terms_count(cv_uni, X_bow_uni, config["top_k"])
    top_bow_bi  = top_terms_count(cv_bi,  X_bow_bi,  config["top_k"])
    top_tf_uni  = top_terms_tfidf(tf_uni, X_tfidf_uni, config["top_k"])
    top_tf_bi   = top_terms_tfidf(tf_bi,  X_tfidf_bi,  config["top_k"])

    output_results(top_bow_uni, "BoW unigram",  f"top_terms/bow_unigram_top{config['top_k']}.csv", save_flag, print_flag)
    output_results(top_bow_bi,  "BoW uni+bi",   f"top_terms/bow_unibi_top{config['top_k']}.csv",  save_flag, print_flag)
    output_results(top_tf_uni,  "TF-IDF unigram", f"top_terms/tfidf_unigram_top{config['top_k']}.csv", save_flag, print_flag)
    output_results(top_tf_bi,   "TF-IDF uni+bi", f"top_terms/tfidf_unibi_top{config['top_k']}.csv",   save_flag, print_flag)

if __name__ == "__main__":
    run_pipeline(save_flag=True, print_flag=False)
