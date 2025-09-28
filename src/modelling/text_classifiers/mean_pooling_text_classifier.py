import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_text_classifier import BaseTextClassifier


class MeanPoolingTextClassifier(BaseTextClassifier):
    """
    A simple text classifier using mean pooling over token embeddings.

    This model embeds input token IDs, computes the mean over token embeddings
    for each document, passes the pooled representation through a feed-forward
    network, and outputs class logits.

    Args:
        vocab_size (int): Size of the vocabulary.
        n_classes (int): Number of target classes.
        embed_dim (int, optional): Dimensionality of token embeddings. Default is 100.
        hidden_dim (int, optional): Number of hidden units in the intermediate layer. Default is 128.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-3.
    """
    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        lr: float = 1e-3
    ) -> None:
        super().__init__(n_classes, lr)
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        self.lr = lr

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (Tensor): Tensor of shape (batch_size, seq_len)
            containing token IDs.

        Returns:
            Tensor: Logits of shape (batch_size, n_classes).
        """
        embedded = self.embedding(input_ids)
        pooled = embedded.mean(dim=1)
        x = F.relu(self.fc1(pooled))
        return self.fc2(x)
