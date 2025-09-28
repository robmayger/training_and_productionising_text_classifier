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

## License

This project is licensed under the [MIT License](LICENSE).

---

For more information, please refer to the [project repository](https://github.com/robmayger/ML-text-classification-project).

--- 
