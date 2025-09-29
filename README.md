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

### Preprocessing

1. **Language detection**
   Detect language of all documents, keep only the documents written in English.

2. **Lowercasing / normalization**  
   Converts all text to lowercase to reduce case-based variability.

3. **Whitespace cleanup**  
   Removes leading/trailing spaces and collapses duplicate whitespace.

4. **Email, phone number, URL removal**
   Strips out emails, links, numbers (e.g. `http://...`, `https://...`, `www...`) using regex.

5. **Punctuation / special character removal**  
   Removes non-alphanumeric symbols to clean up the text.

6. **Tokenization**  
   Splits cleaned text into individual tokens (words).

7. **Stop-word removal**  
   Filters out common words (e.g. *the*, *and*) that add little predictive value.

8. **Stemming / Lemmatization**  
   Reduces words to their base or root form to normalize inflections.

9. **Rejoining / output formatting**  
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

## License

This project is licensed under the [MIT License](LICENSE).

---

For more information, please refer to the [project repository](https://github.com/robmayger/ML-text-classification-project).

--- 
