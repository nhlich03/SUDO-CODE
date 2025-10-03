# deep-learning (Text Summarization with Seq2Seq + Attention)

This project implements a sequence-to-sequence (seq2seq) model with Bahdanau Attention for **Vietnamese text summarization** using the [VietNews dataset](https://huggingface.co/datasets/nam194/vietnews).  

## Structure
```
attention/
  src/
    preprocessing.py
    dataset.py
    train.py
    utils.py
    config.py
  models/
    encoder.py
    decoder.py
    attention.py
    seq2seq.py
  main.py
  requirements.txt
  README.md
```

## Pipeline

1. **Preprocessing**
   - Clean text (lowercasing, whitespace fix).
   - Tokenize `article` and `abstract` with Keras `Tokenizer`.
   - Add `<sos>` and `<eos>` tokens for summaries.
   - Pad sequences to fixed length (`MAX_LEN_CONTENT`, `MAX_LEN_SUMMARY`).

2. **Dataset**
   - `TextSummaryDataset` returns `(content_ids, summary_ids)` tensors.
   - Wrapped in PyTorch `DataLoader`.

3. **Model**
   - **Encoder**: BiGRU with embedding.
   - **Attention**: Bahdanau additive attention over encoder outputs.
   - **Decoder**: GRU with embedding + context vector → Linear → vocab distribution.
   - **Seq2Seq**: teacher-forcing training loop (probability=0.5).

4. **Training**
   - CrossEntropyLoss (ignore padding).
   - Adam optimizer.
   - Gradient clipping to prevent exploding gradients.
   - Logs train & validation loss per epoch.

## Run
1) Install dependencies
```
pip install -r requirements.txt
```

2) Train
```
python main.py
```