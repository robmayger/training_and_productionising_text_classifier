import torch
import torch.nn as nn
from torch import Tensor

from .base_text_classifier import BaseTextClassifier


class TransformerTextClassifier(BaseTextClassifier):
    """
    A text classifier using a Transformer encoder.

    This model embeds input token IDs, adds positional embeddings, passes
    the sequence through a Transformer encoder, and outputs class logits.

    Args:
        vocab_size (int): Size of the vocabulary.
        n_classes (int): Number of target classes.
        embed_dim (int, optional): Dimensionality of token embeddings. Default is 100.
        n_heads (int, optional): Number of attention heads in the Transformer. Default is 4.
        hidden_dim (int, optional): Dimensionality of the feedforward layer in the Transformer. Default is 128.
        n_layers (int, optional): Number of Transformer encoder layers. Default is 2.
        max_len (int, optional): Maximum sequence length. Default is 50.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-3.
    """
    def __init__(
        self, vocab_size: int, n_classes: int, embed_dim: int = 100,
        n_heads: int = 4, hidden_dim: int = 128, n_layers: int = 2,
        max_len: int = 50, lr: float = 1e-3
    ) -> None:
        super().__init__(n_classes, lr)
        self.save_hyperparameters()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc = nn.Linear(embed_dim, n_classes)
        self.lr = lr
        self.max_len = max_len

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (Tensor): Tensor of shape (batch_size, seq_len) containing token IDs.

        Returns:
            Tensor: Logits of shape (batch_size, n_classes).
        """
        batch_size, seq_len = input_ids.size()
        positions = (
            torch.arange(0, seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )

        x = self.embedding(input_ids) + self.pos_embedding(positions)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.fc(x)
