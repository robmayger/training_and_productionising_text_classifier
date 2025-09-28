import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class TransformerTextClassifier(pl.LightningModule):
    def __init__(self, vocab_size, n_classes, embed_dim=100, n_heads=4, hidden_dim=128, n_layers=2, max_len=50, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)  # positional embeddings

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

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.fc(x)

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
