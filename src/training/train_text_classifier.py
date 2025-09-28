from src.modelling import TextDocumentDataset
from src.preprocessing import TextPreprocessor
from src.modelling import MODEL_REGISTRY
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.preprocessing import encode, create_vocab
from torch.utils.data import DataLoader

import pandas as pd
from langdetect import detect, DetectorFactory

import pandas as pd


DetectorFactory.seed = 0

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"


def train_text_classifier(
    df: pd.DataFrame,
    preprocessor: TextPreprocessor,
    config: dict
):

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

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='cpu',
        devices=1
    )

    ModelClass = MODEL_REGISTRY[config["model"]["name"]]

    model = ModelClass(
        vocab_size=vocab_size,
        n_classes = len(le.classes_),
        **config["model"]["params"]
    )

    trainer.fit(model, train_loader, val_loader)

    return model, vocab, le, preprocessor.bigram_model