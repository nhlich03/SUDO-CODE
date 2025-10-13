from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split(csv_path, src_col="en", tgt_col="vi", test_size=0.3, val_ratio_of_test=0.5, seed=42):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[src_col, tgt_col]).reset_index(drop=True)

    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=val_ratio_of_test, random_state=seed)

    train = Dataset.from_pandas(train_df[[src_col, tgt_col]])
    val = Dataset.from_pandas(val_df[[src_col, tgt_col]])
    test = Dataset.from_pandas(test_df[[src_col, tgt_col]])

    return train, val, test