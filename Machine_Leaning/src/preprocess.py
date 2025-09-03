
import re, string
import pandas as pd
from .utils import load_json

def normalize_repeated_letters(text: str) -> str:
    return re.sub(r'([^\W\d_])\1{2,}', r'\1', text, flags=re.UNICODE)

def remove_punct_and_digits(text: str) -> str:
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    text = re.sub(r'\d+', ' ', text)
    text = ''.join(ch for ch in text if not ch.isdigit())
    return text

def replace_slang(text: str, slang_map: dict) -> str:
    if not slang_map:
        return text
    for k, v in slang_map.items():
        text = text.replace(k, v)
    return text

def post_clean(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text(text: str, slang_map: dict = None) -> str:
    if pd.isna(text):
        return None
    text = str(text)
    text = normalize_repeated_letters(text)
    text = remove_punct_and_digits(text)
    text = replace_slang(text, slang_map or {})
    text = post_clean(text)
    if len(text.split()) < 2:
        return None
    return text

def apply_clean(df, text_col: str, replace_list_path: str = None):
    df = df.copy()
    slang_map = load_json(replace_list_path) if replace_list_path else {}
    df[text_col] = df[text_col].apply(lambda x: clean_text(x, slang_map=slang_map))
    df = df.dropna(subset=[text_col])
    return df
