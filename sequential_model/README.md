# Vietnamese Books LSTM LM (Skeleton)

A clean, modular PyTorch project to train a language model on the "10000 Vietnamese Books" dataset from Kaggle.

---

## Project Structure
```
.
├── config
│   └── config.yaml           # all hyperparameters & settings
├── output
│   └── checkpoints/          # best model checkpoint
├── src
│   ├── data.py               # preprocessing, vocab, dataset, dataloader
│   ├── model.py              # LSTM language model
│   ├── train.py              # training loop, evaluation
│   ├── utils.py              # config loader, seeding, helpers
│   ├── main.py               # entrypoint: train + validate + test
│   └── generate.py           # inference: generate text from prompt
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Train
```bash
python -m src.main --config config/config.yaml
```

- The best checkpoint will be saved to:  
  `output/checkpoints/best.pt`
- Vocabulary is saved at:  
  `output/vocab.json`
- Training history (loss & perplexity) is logged to:  
  `output/history.json`

---

## Generate
```bash
python -m src.generate --config config/config.yaml --prompt "<BOS> " --max_new_tokens 200
```

---

## Model Usage

After training:

- **Input**: A starting text prompt.  
  Example:
  ```bash
  python -m src.generate --config config/config.yaml --prompt "Ngày xửa ngày xưa" --max_new_tokens 50
  ```

- **Output**: The model generates a continuation of the text based on the prompt.  
  Example result:
  ```
  <BOS> Ngày xửa ngày xưa có một cậu bé sống trong ngôi làng nhỏ ...
  ```

- **Parameters**:
  - `--max_new_tokens`: maximum number of tokens to generate.
  - `--temperature`: controls randomness (higher → more diverse text).
  - `--top_k`: sample only from the top k tokens.
  - `--top_p`: nucleus sampling, sample from tokens with cumulative probability ≥ p.

---

## Notes
- The dataset is automatically downloaded with [kagglehub](https://github.com/Kaggle/kagglehub).  
- The model is a simple LSTM-based language model, not optimized for production.  
- Future improvements may include Transformer-based models and more advanced tokenization.

---
