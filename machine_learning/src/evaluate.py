
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import json

def evaluate_and_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_test, y_pred)
    return report, acc, y_pred

def save_predictions(ids, texts, y_true, y_pred, out_csv_path: str):
    df = pd.DataFrame({
        "id": ids if ids is not None else range(len(texts)),
        "text": texts,
        "true": y_true,
        "pred": y_pred,
    })
    df.to_csv(out_csv_path, index=False)
    return out_csv_path

def save_metrics(report: dict, acc: float, out_path_csv: str, out_path_json: str = None):
    df = pd.DataFrame(report).T  # transpose để labels thành rows
    df.loc["accuracy"] = [None, None, acc, None]  # thêm accuracy riêng
    df.to_csv(out_path_csv)

    if out_path_json:
        with open(out_path_json, "w", encoding="utf-8") as f:
            json.dump({"accuracy": acc, "report": report}, f, ensure_ascii=False, indent=2)

    return out_path_csv, out_path_json