
import pandas as pd

def load_datasets(train_path: str, test_path: str, text_col: str, label_col: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Keep only necessary columns
    train_df = train_df[[text_col, label_col]].dropna()
    test_df = test_df[[text_col, label_col]].dropna()

    return train_df, test_df
