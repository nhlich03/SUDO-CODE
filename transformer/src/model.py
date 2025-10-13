from transformers import MarianTokenizer, MarianMTModel
import torch

class TranslationModel:
    def __init__(self, model_name, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device)

    def save(self, dirpath):
        self.model.save_pretrained(dirpath)
        self.tokenizer.save_pretrained(dirpath)


    def translate(self, texts, max_length=128):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in outputs]