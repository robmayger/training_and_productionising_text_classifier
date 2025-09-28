from pathlib import Path
import os
import pandas as pd
import yaml

import tempfile
import mlflow
import mlflow.pytorch
import joblib
from mlflow.tracking import MlflowClient

from src.preprocessing import TextPreprocessor
from src.training.train_text_classifier import train_text_classifier


def train_and_log_model(df, config):

    preprocessor = TextPreprocessor(use_spell_correction=False)

    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run() as run:

        model, vocab, le, bigram_model = train_text_classifier(df, preprocessor, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save artifacts in a subdirectory
            artifacts_dir = os.path.join(tmpdir, "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)

            joblib.dump(vocab, os.path.join(artifacts_dir, "vocab.pkl"))
            joblib.dump(le, os.path.join(artifacts_dir, "label_encoder.pkl"))
            joblib.dump(bigram_model, os.path.join(artifacts_dir, "bigram_model.pkl"))

            # Log the model with extra artifacts
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name=config["model"]["name"],
                extra_files=[os.path.join(artifacts_dir, f) for f in os.listdir(artifacts_dir)]
            )

        # Log metrics & params
        mlflow.log_param("model_name", config["model"]["name"])
        mlflow.log_param("vocab_size", len(vocab))

        run_id = run.info.run_id
        print("Logged Run ID:", run_id)
    
    return run_id


def promote_model(config):

    client = MlflowClient()

    latest_version = client.get_latest_versions(config["model"]["name"])[0].version

    client.transition_model_version_stage(
        name=config["model"]["name"],
        version=latest_version,
        stage="Production"
    )


if __name__ == "__main__":

    base_path = Path(os.getcwd())

    config_path = base_path / "src" / "config" / "config.yml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    print(config)

    df = (
        pd.read_csv(base_path / config["data_file_path"])
        .filter(['id', 'company_description', 'source', 'is_edited', 'created_at'])
        .dropna()
        .astype(
            {
                "id": "int64",
                "is_edited": "float64"
            }
        )
        .assign(created_at=lambda d: pd.to_datetime(d["created_at"]))
    )

    run_id = train_and_log_model(df, config)

    promote_model(config)

