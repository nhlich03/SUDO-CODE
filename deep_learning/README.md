# Text Classification with PyTorch (Vietnamese 10 Topics)

This project implements a simple **text classification** pipeline using **PyTorch** on the [VNTC dataset](https://github.com/duyvuleo/VNTC).  
The dataset contains Vietnamese news articles categorized into 10 different topics.

---

## Project Structure

```
.
├── deep_learning.ipynb   # Main notebook (preprocessing, training, evaluation)
├── vocab.json            # Saved vocabulary (optional)
├── best_model.pt         # Saved best model (checkpoint)
└── README.md
```

---

## Features
- Vietnamese text preprocessing:
  - Lowercasing, regex normalization for numbers (`1k → 1 ngàn`, `50% → 50 phần trăm`, etc.)
  - Remove special characters
  - Tokenization with simple `.split()` (or can replace with Vietnamese tokenizer)
  - Stopword removal
- Vocabulary building with frequency cutoff (`min_freq`, `max_vocab`)
- Custom `Dataset` and `DataLoader` with padding and batching
- Simple neural model:
  - Embedding → Average pooling → Fully connected layers
  - Dropout & ReLU
- Training pipeline with:
  - CrossEntropyLoss + Adam optimizer
  - Early stopping based on validation accuracy
  - Model checkpointing
- Evaluation with:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
  - Training/Validation loss & accuracy curves

---

## Dataset
We use the **VNTC - Vietnamese News Text Classification** dataset.  
- **10 Topics**:
  - Chính trị Xã hội
  - Đời sống
  - Khoa học
  - Kinh doanh
  - Pháp luật
  - Sức khỏe
  - Thế giới
  - Thể thao
  - Văn hóa
  - Vi tính  

- Train size: ~30,000 samples  
- Test size: ~50,000 samples  

---

## Results

- **Test Accuracy**: ~89.7%  
- **Macro F1-score**: ~0.87  
- **Weighted F1-score**: ~0.89  

### Strongest classes:
- Thể thao, Vi tính, Thế giới (>92% F1)

### Hardest classes:
- Đời sống, Khoa học (<77% F1)

---

## Future Improvements
- Use pretrained Vietnamese embeddings (fastText, PhoBERT, viBERT).  
- Apply deeper architectures (CNN, LSTM, Transformer).  
- Data augmentation or rebalancing for underperforming classes.  
- More advanced tokenization for Vietnamese (e.g. `underthesea`, `pyvi`).  
 
