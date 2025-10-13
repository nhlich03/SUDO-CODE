from model import TranslationModel
from utils import load_config

def main(model_dir=None, sentences=None):
    cfg = load_config()
    model_dir = model_dir or cfg["train"]["output_dir"]

    tm = TranslationModel(model_dir)

    if isinstance(sentences, str):
        sentences = [sentences]

    outputs = tm.translate(sentences)
    for s, t in zip(sentences, outputs):
        print("EN:", s)
        print("VI:", t)
        print()




if __name__ == "__main__":
    main(sentences=["I love machine learning.", "How are you?"])