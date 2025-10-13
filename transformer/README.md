# machine-translation-transformer (English → Vietnamese, Hugging Face Transformers)

This is your notebook split into `.py` files for fine-tuning a Transformer-based machine translation model (English → Vietnamese) using the Hugging Face `MarianMT` architecture.

---

## Structure
```
machine-translation-transformer/
    src/
        model.py
        dataset.py
        train.py
        infer.py
        utils.py
    configs/
        config.yaml
    data/
        data.csv
    requirements.txt
    README.md
    .gitignore
```

---

## Pipeline
1. **Data loading & splitting** in `dataset.py`
   - Load CSV (English–Vietnamese)
   - Split into train / val / test sets  
   - Tokenization via pretrained Marian tokenizer  

2. **Preprocessing**
   - Truncation & padding handled automatically during tokenization
   - Convert text → token IDs for both source and target  

3. **Model**
   - Load pretrained `Helsinki-NLP/opus-mt-en-vi`  
   - Fine-tune with `Seq2SeqTrainer`  

4. **Training**
   - Defined in `train.py`  
   - BLEU evaluation after each epoch  
   - Best checkpoints saved to `output/checkpoints/`

5. **Inference**
   - Run translation on new sentences or files via `infer.py`  
   - Save outputs to `output/translations/`

---


