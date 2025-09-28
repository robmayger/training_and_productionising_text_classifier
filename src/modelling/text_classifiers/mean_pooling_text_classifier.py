import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F


class MeanPoolingTextClassifier(pl.LightningModule):
    def __init__(self, vocab_size, n_classes, embed_dim=100, hidden_dim=128, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        self.lr = lr

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        pooled = embedded.mean(dim=1)
        x = F.relu(self.fc1(pooled))
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'])
        loss = F.cross_entropy(logits, batch['labels'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['input_ids'])
        loss = F.cross_entropy(logits, batch['labels'])
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch['labels']).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)