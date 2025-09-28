from torch.utils.data import Dataset
import torch


class TextDocumentDataset(Dataset):
    def __init__(self, documents: list[list[int]], labels: list[int], max_len=50):
        self.documents = documents
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        ids = self.documents[idx]
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }