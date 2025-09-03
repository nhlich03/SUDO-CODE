
from typing import Dict, Any
from .data import load_datasets
from .preprocess import apply_clean
from .models import build_pipeline
from .evaluate import evaluate_and_report, save_predictions
from .utils import save_model, ensure_dir

def run_training(cfg: Dict[str, Any]):
    # Unpack config with defaults
    train_path = cfg.get("train_path")
    test_path = cfg.get("test_path")
    text_col = cfg.get("text_col", "comment")
    label_col = cfg.get("label_col", "label")
    model_name = cfg.get("model", "svm")
    tfidf = cfg.get("tfidf", {}) or {}
    ngram_min = tfidf.get("ngram_min", 1)
    ngram_max = tfidf.get("ngram_max", 2)
    min_df = tfidf.get("min_df", 2)
    max_df = tfidf.get("max_df", 0.9)

    artifacts_dir = cfg.get("artifacts_dir", "artifacts")
    outputs_dir = cfg.get("outputs_dir", "outputs")
    custom_model_name = cfg.get("model_name")
    replace_list_path = cfg.get("replace_list_path")

    train_df, test_df = load_datasets(train_path, test_path, text_col, label_col)

    train_df = apply_clean(train_df, text_col, replace_list_path=replace_list_path)
    test_df  = apply_clean(test_df,  text_col, replace_list_path=replace_list_path)

    X_train, y_train = train_df[text_col], train_df[label_col]
    X_test,  y_test  = test_df[text_col],  test_df[label_col]

    pipe = build_pipeline(
        model=model_name,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        min_df=min_df,
        max_df=max_df
    )
    pipe.fit(X_train, y_train)

    report, acc, y_pred = evaluate_and_report(pipe, X_test, y_test)

    # Save artifacts
    ensure_dir(artifacts_dir)
    ensure_dir(outputs_dir)
    base_name = custom_model_name if custom_model_name else model_name
    model_file = f"{artifacts_dir}/model_{base_name}.pkl"
    save_model(pipe, model_file)

    preds_file = f"{outputs_dir}/predictions_{base_name}.csv"
    save_predictions(ids=None, texts=X_test.tolist(), y_true=y_test.tolist(), y_pred=y_pred, out_csv_path=preds_file)

    metrics_csv = f"{outputs_dir}/metrics_{base_name}.csv"
    metrics_json = f"{outputs_dir}/metrics_{base_name}.json"

    from .evaluate import save_metrics
    save_metrics(report, acc, metrics_csv, metrics_json)

    return {
        "accuracy": acc,
        "report": report,
        "model_path": model_file,
        "predictions_csv": preds_file,
        "metrics_csv": metrics_csv,
        "metrics_json": metrics_json
    }