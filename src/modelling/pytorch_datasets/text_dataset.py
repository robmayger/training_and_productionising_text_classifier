from typing import List, Dict, Any
from torch.utils.data import Dataset
import torch


class TextDocumentDataset(Dataset):
    """
    PyTorch Dataset for text documents represented as sequences of token IDs.

    Each document is a list of integers representing token IDs. Labels are 
    integer class labels corresponding to each document.

    Args:
        documents (List[List[int]]): List of documents, where each document is a list of token IDs.
        labels (List[int]): List of integer labels corresponding to each document.
        max_len (int, optional): Maximum length for the documents. Default is 50.

    Attributes:
        documents (List[List[int]]): Stored list of documents.
        labels (List[int]): Stored list of labels.
        max_len (int): Maximum length for document sequences.
    """
    def __init__(
        self,
        documents: List[List[int]],
        labels: List[int],
        max_len: int = 50
    ) -> None:
        self.documents = documents
        self.labels = labels
        self.max_len = max_len

    def __len__(self) -> int:
        """
        Returns:
            int: Number of documents in the dataset.
        """
        return len(self.documents)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single document and its label as tensors.

        Args:
            idx (int): Index of the document to retrieve.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'input_ids': Tensor of token IDs (padded/truncated if necessary).
                - 'labels': Tensor of the corresponding label.
        """
        ids = self.documents[idx]
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
