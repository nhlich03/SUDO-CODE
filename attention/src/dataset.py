import torch
from torch.utils.data import Dataset

class TextSummaryDataset(Dataset):
    def __init__(self, contents, summaries):
        self.contents = contents
        self.summaries = summaries

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        content = torch.tensor(self.contents[idx], dtype=torch.long)
        summary = torch.tensor(self.summaries[idx], dtype=torch.long)
        return content, summary
