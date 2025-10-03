import time
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import pandas as pd

from src.preprocessing import Preprocessor
from src.dataset import TextSummaryDataset
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq
from src.train import train, evaluate
from src import config
from torch.utils.data import DataLoader
from src.utils import epoch_time

# Load dataset
ds = load_dataset("nam194/vietnews")
train_df = pd.DataFrame(ds['train'])
val_df = pd.DataFrame(ds['validation'])
test_df = pd.DataFrame(ds['test'])

contents = train_df["article"].astype(str).tolist()
summaries = train_df["abstract"].astype(str).tolist()

pre = Preprocessor(max_len_content=config.MAX_LEN_CONTENT, max_len_summary=config.MAX_LEN_SUMMARY)
pre.fit_tokenizers(contents, summaries)

X_train, y_train = pre.transform(train_df['article'], train_df['abstract'])
X_val, y_val = pre.transform(val_df['article'], val_df['abstract'])
X_test, y_test = pre.transform(test_df['article'], test_df['abstract'])

content_vocab_size, summary_vocab_size = pre.get_vocab_size()

# DataLoader
train_dataset = TextSummaryDataset(X_train, y_train)
val_dataset = TextSummaryDataset(X_val, y_val)
test_dataset = TextSummaryDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(content_vocab_size, config.EMBED_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS)
decoder = Decoder(summary_vocab_size, config.EMBED_SIZE, config.HIDDEN_SIZE)
model = Seq2Seq(encoder, decoder, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=config.LR)

# Training
for epoch in range(config.N_EPOCHS):
    start_time = time.time()
    train_loss = train(model, train_loader, optimizer, criterion)
    valid_loss = evaluate(model, val_loader, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f"Epoch {epoch+1}/{config.N_EPOCHS} | Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f}")
