from src.modelling import TextDocumentDataset
from src.preprocessing import TextPreprocessor
from src.modelling import MODEL_REGISTRY
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.preprocessing import encode, create_vocab
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Tuple, Dict, Any

import pandas as pd
from langdetect import detect, DetectorFactory

import pandas as pd


DetectorFactory.seed = 0

def detect_language(text: str) -> str:
    """
    Detect the language of a given text.

    Args:
        text (str): Input text string.

    Returns:
        str: ISO language code (e.g., 'en') or 'unknown' if detection fails.
    """
    try:
        return detect(text)
    except:
        return "unknown"


def train_text_classifier(
    df: pd.DataFrame,
    preprocessor: TextPreprocessor,
    config: Dict[str, Any]
) -> Tuple[pl.LightningModule, Dict[str, int], LabelEncoder, Any]:
    """
    Train a text classifier using PyTorch Lightning and preprocess the data.

    Args:
        df (pd.DataFrame): DataFrame containing 'company_description' and 'source' columns.
        preprocessor (TextPreprocessor): Preprocessor instance for cleaning, tokenizing, and building bigrams.
        config (Dict[str, Any]): Configuration dictionary with model and training parameters.

    Returns:
        Tuple containing:
            - model (pl.LightningModule): Trained PyTorch Lightning model.
            - vocab (Dict[str, int]): Vocabulary mapping tokens to indices.
            - le (LabelEncoder): LabelEncoder fitted on the target labels.
            - bigram_model (Any): Trained bigram model from the preprocessor.
    """

    df['language'] = df['company_description'].apply(detect_language)
    df = df[df.language == config['language']]

    token_lists = [preprocessor.tokenize(preprocessor.clean_text(desc)) for desc in df['company_description']]
    preprocessor.build_bigrams(token_lists)

    df = df.copy()
    df['clean'] = df['company_description'].apply(preprocessor.preprocess)

    le = LabelEncoder()
    df['target'] = le.fit_transform(df['source'])

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['target'], test_size=0.2, random_state=2025, stratify=df['target']
    )

    vocab = create_vocab(X_train)
    vocab_size = len(vocab) + 1

    train_encodings = list(map(lambda x: encode(x, vocab), X_train))
    test_encodings = list(map(lambda x: encode(x, vocab), X_test))

    train_dataset = TextDocumentDataset(train_encodings, y_train.tolist())
    val_dataset = TextDocumentDataset(test_encodings, y_test.tolist())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        filename="best-model",
    )

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='cpu',
        devices=1,
        callbacks=[checkpoint_callback]
    )

    ModelClass = MODEL_REGISTRY[config["model"]["name"]]

    model = ModelClass(
        vocab_size=vocab_size,
        n_classes=len(le.classes_),
        **config["model"]["params"]
    )

    trainer.fit(model, train_loader, val_loader)

    best_model_path = checkpoint_callback.best_model_path
    best_f1 = checkpoint_callback.best_model_score

    print()
    print(f"Best model saved at: {best_model_path}")
    print(f"Best validation F1: {best_f1.item():.4f}")
    print()

    best_model = ModelClass.load_from_checkpoint(best_model_path)

    return best_model, vocab, le, preprocessor.bigrams