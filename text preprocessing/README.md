# Vietnamese Text Processing Project

This project implements a **text preprocessing and vectorization pipeline** for Vietnamese online news articles.

## Overview

The workflow includes:

- **Dataset download** from Kaggle (`haitranquangofficial/vietnamese-online-news-dataset`).  
- **Text preprocessing**: cleaning, Vietnamese normalization, tokenization, stopword removal, optional lemmatization.  
- **Vectorization**:  
  - Bag-of-Words (BoW) with unigram and bigram features.  
  - TF-IDF with unigram and bigram features.  
- **Reporting**:  
  - Corpus statistics (average tokens, min/max tokens).  
  - Representation stats (vocab size, sparsity).  
  - Top frequent terms (BoW) and top weighted terms (TF-IDF).  
- Results can be printed to console or exported to CSV files in the `results/` folder.

## Project Structure

```
.
├─ config.py           # Configurable parameters (dataset id, sample size, vectorizer settings)
├─ main.py             # Entry point to run the pipeline
├─ utils.py            # I/O helpers, stopword loader, output_results
├─ preprocessing.py    # Text cleaning and tokenization functions
├─ features.py         # Vectorizer builders and reporting utilities
├─ stop_words.txt      # Vietnamese stopwords list
├─ results/            # Output CSVs (if save_flag=True)
└─ README.md
```

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the pipeline:
   ```bash
   python main.py
   ```

3. To save results as CSV files in `results/`, set `save_flag=True` in `main.py`:
   ```python
   if __name__ == "__main__":
       run_pipeline(save_flag=True)
   ```

## Outputs

When `save_flag=True`, the following CSVs will be written into `results/`:

- `corpus_stats.csv` – corpus statistics.  
- `representations.csv` – vocab size, sparsity of each representation.  
- `top_terms/bow_unigram_topK.csv`  
- `top_terms/bow_unibi_topK.csv`  
- `top_terms/tfidf_unigram_topK.csv`  
- `top_terms/tfidf_unibi_topK.csv`

## Notes

- By default only **5,000 sample rows** are used (configurable in `config.py`).  
- Adjust `min_df`, `max_df`, and `ngram_range` in `config.py` to tune vocabulary size.  
- Stopwords can be updated in `stop_words.txt`.  
- Empty rows after preprocessing may appear (min_tokens = 0). You can filter them depending on downstream tasks.  
