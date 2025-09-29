# Training and Productionising Text Classifier

This project demonstrates how to train and deploy a text classification model using Docker. It provides a reproducible environment for training a model and serving it as a REST API endpoint, facilitating integration into production systems.

## Features

- Controlled via configuration file in `src/config` directory.
- Modular codebase organized within the `src/` directory.
- Support for training and evaluating various classification models.
- Dockerized environment for consistent setup and deployment.
- Jupyter Notebooks for exploratory data analysis and prototyping.

## Project Structure

```
.
├── .devcontainer/           # Development container configuration
├── src/                     # Source code for data processing and modeling
├── notebooks/               # Jupyter Notebooks for EDA and experiments
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker image definition
├── docker-compose.yml       # Docker Compose configuration
├── README.md                # Project documentation
```

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed on your system.

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/robmayger/training_and_productionising_text_classifier.git
   cd training_and_productionising_text_classifier
   ```


2. **Build and run the Docker container:**

   ```bash
   docker-compose up --build
   ```


   This will set up the development environment and install all necessary dependencies.

3. **Train the classification model:**

   ```bash
   python3 src/tech_test_2025/main.py
   ```

4. **Run the FastAPI app:**

   ```bash
   uvicorn src.tech_test_2025.app:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Query the model:**

   ```bash
   curl -X POST "http://127.0.0.1:1111/predict" \
   -H "Content-Type: application/json" \
   -d '{"text": "Example text to classify"}'
   ```

## Approach

### Configuration and Reproducibility

The entire modeling and deployment pipeline is **governed by a configuration file** (`config.yml`). This ensures that:

- **Reproducibility**: Experiments can be rerun with the same settings, producing consistent results.  
- **Hyperparameter tuning**: Model hyperparameters (e.g., embedding size, hidden dimensions, number of layers, learning rate) can be easily modified through the config file to improve prediction accuracy without changing the code.  
- **Deployment control**: MLflow experiment names, model registry settings, and artifact paths are all configured centrally, ensuring smooth promotion from training to Production.

With the current setup, the pipeline achieves a **mean F1 score of 0.9** on the validation dataset, providing a strong baseline for text classification tasks.

### Preprocessing

1. **Language detection**
   Detect language of all documents, keep only the documents written in English.

2. **Spelling correction**
   Automated spelling correction for text documents. This is disabled by default.

3. **Lowercasing / normalization**  
   Converts all text to lowercase to reduce case-based variability.

4. **Whitespace cleanup**  
   Removes leading/trailing spaces and collapses duplicate whitespace.

5. **Email, phone number, URL removal**
   Strips out emails, links, numbers (e.g. `http://...`, `https://...`, `www...`) using regex.

6. **Punctuation / special character removal**  
   Removes non-alphanumeric symbols to clean up the text.

7. **Tokenization**  
   Splits cleaned text into individual tokens (words).

8. **Stop-word removal**  
   Filters out common words (e.g. *the*, *and*) that add little predictive value.

9. **Stemming / Lemmatization**  
   Reduces words to their base or root form to normalize inflections.

10. **Rejoining / output formatting**  
   Returns the processed tokens as either a list or a cleaned string for downstream tasks.

### Text Classifier Models

In `src/modelling/text_classifiers/` there are two neural text classifiers implemented:

#### 1. MeanPoolingTextClassifier
A simple feed-forward model that averages token embeddings to form a document representation.

- **Architecture**:
  - Embedding layer for tokens
  - Mean pooling across the sequence dimension
  - Feed-forward network with one hidden layer
  - Output layer projecting to class logits
- **Key Args**:
  - `vocab_size` (int): Vocabulary size
  - `n_classes` (int): Number of output classes
  - `embed_dim` (default: 100): Token embedding size
  - `hidden_dim` (default: 128): Hidden layer size
  - `lr` (default: 1e-3): Learning rate
- **Strengths**: Fast, lightweight, interpretable baseline.
- **Limitations**: Loses word order information.

---

#### 2. TransformerTextClassifier
A Transformer-based classifier that leverages self-attention to model contextual relationships between tokens.

- **Architecture**:
  - Embedding layer + positional embeddings
  - Transformer encoder stack (multi-head self-attention + feedforward layers)
  - Mean pooling over sequence outputs
  - Linear output layer projecting to class logits
- **Key Args**:
  - `vocab_size` (int): Vocabulary size
  - `n_classes` (int): Number of output classes
  - `embed_dim` (default: 100): Token embedding size
  - `n_heads` (default: 4): Number of attention heads
  - `hidden_dim` (default: 128): Feedforward hidden dimension
  - `n_layers` (default: 2): Number of Transformer layers
  - `max_len` (default: 50): Maximum sequence length
  - `lr` (default: 1e-3): Learning rate
- **Strengths**: Captures word order and contextual dependencies, more expressive than mean pooling.
- **Limitations**: Heavier computational cost, requires more data to generalize well.

### Model Tracking and Deployment

This project uses **MLflow** to manage the lifecycle of trained text classifiers.  
Models, vocabularies, encoders, and other preprocessing artifacts are logged during training, then promoted and served via FastAPI.

---

#### Training and Tracking

The `train_and_log_model` function:

1. **Preprocessing**  
   - A `TextPreprocessor` cleans and prepares input text.  

2. **Training**  
   - A classifier is trained via `train_text_classifier`.

3. **Artifact Management**  
   - Supporting artifacts (vocabulary, label encoder, bigram model) are serialized with `joblib` and stored temporarily.

4. **MLflow Logging**  
   - The trained PyTorch model is logged with `mlflow.pytorch.log_model`.
   - Extra artifacts (e.g. `vocab.pkl`, `label_encoder.pkl`, `bigram_model.pkl`) are attached to the run.
   - Model and experiment parameters (e.g. `model_name`, `vocab_size`) are logged for reproducibility.

5. **Experiment Tracking**  
   - Each run is tracked under the configured MLflow experiment.
   - The run ID is printed for reference.

---

#### Model Promotion

The `promote_model` function:

- Uses the **MLflow Model Registry** to find the latest registered version of the trained model.
- Transitions that version to the **`Production`** stage, making it the canonical model for deployment.

---

#### Deployment with FastAPI

The deployed API loads the **Production model** and its artifacts from MLflow:

1. **Loading**  
   - `mlflow.pytorch.load_model` loads the trained classifier from the registry.
   - Associated artifacts (vocab, label encoder, bigram model) are downloaded and loaded via `joblib`.

2. **Preprocessing**  
   - A `TextPreprocessor` applies the same cleaning steps as during training, using the loaded bigram model.

3. **Language Detection**  
   - Input text is first checked with `langdetect`; non-English input is rejected.

4. **Prediction**  
   - The text is preprocessed and encoded with the stored vocabulary.
   - The model produces class probabilities via PyTorch.
   - The predicted class index is mapped back to a label using the label encoder.

5. **API Response**  
   - The `/predict` endpoint returns:
     - Original input text  
     - Predicted class index  
     - Predicted class label  
     - Probability distribution across all classes

---

## Future work

### 1. Model Architecture
- **Additional architectures**: Experiment with model architectures such as CNNs, RNNs (LSTM/GRU), or pretrained transformers like BERT or RoBERTa for potentially higher accuracy.
- **Other model types**: Explore traditional machine learning models like Logistic Regression, Naive Bayes, Support Vector Machines (SVM), Random Forests, or Gradient Boosted Trees. These can be effective for smaller datasets or when computational resources are limited.
- **Hyperparameter tuning**: Automate hyperparameter search (e.g., learning rate, hidden dimensions, number of layers, dropout) using libraries like Optuna or Ray Tune.

### 2. Preprocessing Enhancements
- **Spell correction**: Enable and evaluate spell correction in `TextPreprocessor`.
- **More sophisticated tokenization**: Incorporate subword tokenization (e.g., SentencePiece, WordPiece) to better handle rare words.
- **Data augmentation**: Apply techniques like synonym replacement, back-translation, or paraphrasing to enrich the training data.

### 3. Feature Engineering
- **Contextual embeddings**: Use pretrained embeddings (GloVe, FastText) or contextual embeddings from transformers to capture richer semantic information.
- **Additional features**: Incorporate metadata (e.g., source, timestamp) or n-gram features for improved predictive power.

### 4. Model Tracking and Experimentation
- **Automated metrics logging**: Extend the use of MLflow to track more metrics. Log metrics in each training epoch.
- **Version comparison**: Implement dashboards or automated reports to compare different model versions and configurations in MLflow.
- **Model Promotion**: Implement proper model promotion logic.

### 5. Deployment and API
- **Multi-language support**: Extend language detection and preprocessing to handle non-English text.
- **Batch predictions**: Support batch prediction endpoints to efficiently handle multiple documents at once.
- **Monitoring and logging**: Track prediction latency, input statistics, and model drift over time.

### 6. Scalability and Performance
- **GPU support**: Leverage GPU acceleration for both training and inference for faster performance.
- **Optimized inference**: Use techniques like model quantization or ONNX export to reduce memory usage and improve prediction speed.

### 7. Testing and Quality Assurance
- **Unit tests**: Implement tests for preprocessing, encoding, and model outputs to ensure robustness.
- **Continuous integration**: Set up CI/CD pipelines to automatically validate and deploy new model versions.

---

This setup ensures that:
- Training and inference use **the exact same preprocessing and artifacts**.
- All experiments are reproducible and tracked in **MLflow**.
- Deployment always serves the **latest Production model** with FastAPI.

## License

This project is licensed under the [MIT License](LICENSE).

---

For more information, please refer to the [project repository](https://github.com/robmayger/ML-text-classification-project).

--- 
