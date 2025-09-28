from pathlib import Path
import os
import yaml
import mlflow
import mlflow.pytorch
import joblib
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.preprocessing import TextPreprocessor
from src.preprocessing import encode
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0


def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"


base_path = Path(os.getcwd())

config_path = base_path / "src" / "config" / "config.yml"

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

model = mlflow.pytorch.load_model(f"models:/{config["model"]["name"]}/Production")
vocab = joblib.load(
    mlflow.artifacts.download_artifacts(
        f"models:/{config["model"]["name"]}/Production/extra_files/vocab.pkl"
    )
)
bigram_model = joblib.load(
    mlflow.artifacts.download_artifacts(
        f"models:/{config["model"]["name"]}/Production/extra_files/bigram_model.pkl"
    )
)
le = joblib.load(
    mlflow.artifacts.download_artifacts(
        f"models:/{config["model"]["name"]}/Production/extra_files/label_encoder.pkl"
    )
)

preprocessor = TextPreprocessor(use_spell_correction=False)
preprocessor.bigrams = bigram_model

# ---- FastAPI App ----
app = FastAPI()

class TextIn(BaseModel):
    text: str

@app.post("/predict")
def predict(inp: TextIn):

    if detect_language(inp.text) != "en":
        raise HTTPException(
            status_code=400,
            detail="Current model does not support non-English text."
        )


    clean = preprocessor.preprocess(inp.text)
    encoded = encode(clean, vocab)

    x = torch.tensor([encoded])

    model.eval()
    with torch.no_grad():
        probs = model(x)
        predicted_class = torch.argmax(probs, dim=-1).item()
        predicted_label = le.inverse_transform([predicted_class])[0]

    return {
        "text": inp.text,
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "probabilities": probs.tolist()
    }
