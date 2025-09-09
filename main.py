import yaml
from pathlib import Path
from dataset import load_texts
from preprocess import preprocess_corpus
from embedding import train_word2vec, save_model
from visualize import plot_embeddings

if __name__ == "__main__":
    # Load config
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))

    # Load & preprocess
    texts = load_texts(cfg["data"]["input_path"], cfg["data"]["text_column"])
    sents = preprocess_corpus(texts, use_lemmatize=cfg["preprocess"].get("use_lemmatize", True))

    # Train
    model = train_word2vec(sents, **cfg["train"])
    Path("models").mkdir(exist_ok=True)
    save_model(model, "models/w2v.model")

    # Visualize
    vis_cfg = cfg.get("visualize", {})
    if "output_path" in vis_cfg:
        Path(vis_cfg["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        plot_embeddings(model, **vis_cfg)
