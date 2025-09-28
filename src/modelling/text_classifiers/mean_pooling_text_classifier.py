import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict


class MeanPoolingTextClassifier(pl.LightningModule):
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
        super().__init__()
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

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        Training step for PyTorch Lightning.

        Args:
            batch (Dict[str, Tensor]): Batch dictionary with keys 'input_ids' and 'labels'.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Training loss.
        """
        logits = self(batch['input_ids'])
        loss = F.cross_entropy(logits, batch['labels'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """
        Validation step for PyTorch Lightning.

        Args:
            batch (Dict[str, Tensor]): Batch dictionary with keys 'input_ids' and 'labels'.
            batch_idx (int): Index of the batch.
        """
        logits = self(batch['input_ids'])
        loss = F.cross_entropy(logits, batch['labels'])
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch['labels']).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: The Adam optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
