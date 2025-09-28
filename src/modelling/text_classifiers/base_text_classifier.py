import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torch import Tensor
from typing import Dict


class BaseTextClassifier(pl.LightningModule):
    """
    Base class for text classifiers in PyTorch Lightning.

    This class defines common training and validation logic, including
    metrics suitable for imbalanced classification (macro-averaged
    precision, recall, F1 score, accuracy, and balanced accuracy).
    Subclasses must implement the `forward` method to define the model
    architecture.

    Args:
        n_classes (int): Number of target classes.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-3.
    """
    def __init__(self, n_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        # ⚡ Shared metrics
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=n_classes, average="macro"
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=n_classes, average="macro"
        )
        self.val_precision = torchmetrics.Precision(
            task="multiclass", num_classes=n_classes, average="macro"
        )
        self.val_recall = torchmetrics.Recall(
            task="multiclass", num_classes=n_classes, average="macro"
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=n_classes, average="macro"
        )
        self.val_bal_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=n_classes, average="macro"
        )

        self.lr = lr
        self.n_classes = n_classes

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (Tensor): Tensor of shape (batch_size, seq_len) 
                containing token IDs.

        Returns:
            Tensor: Logits of shape (batch_size, n_classes).
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def compute_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Compute the loss function for classification.

        Args:
            logits (Tensor): Predicted logits of shape (batch_size, n_classes).
            labels (Tensor): Ground truth labels of shape (batch_size,).

        Returns:
            Tensor: Cross-entropy loss.
        """
        return F.cross_entropy(logits, labels)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        Training step.

        Args:
            batch (Dict[str, Tensor]): Dictionary with keys:
                - "input_ids": Tensor of token IDs.
                - "labels": Tensor of target class indices.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Training loss for the batch.
        """
        logits = self(batch["input_ids"])
        loss = self.compute_loss(logits, batch["labels"])
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, batch["labels"])
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """
        Validation step.

        Args:
            batch (Dict[str, Tensor]): Dictionary with keys:
                - "input_ids": Tensor of token IDs.
                - "labels": Tensor of target class indices.
            batch_idx (int): Index of the current batch.
        """
        logits = self(batch["input_ids"])
        loss = self.compute_loss(logits, batch["labels"])
        preds = torch.argmax(logits, dim=1)

        self.val_acc.update(preds, batch["labels"])
        self.val_precision.update(preds, batch["labels"])
        self.val_recall.update(preds, batch["labels"])
        self.val_f1.update(preds, batch["labels"])
        self.val_bal_acc.update(preds, batch["labels"])

        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        Compute and log validation metrics at the end of an epoch.
        Resets metric states afterwards to avoid accumulation.
        """
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_precision", self.val_precision.compute())
        self.log("val_recall", self.val_recall.compute())
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_bal_acc", self.val_bal_acc.compute())

        # Reset states
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_bal_acc.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer.

        Returns:
            torch.optim.Optimizer: Adam optimizer with the configured learning rate.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
