# AI Lead – Technical Assessment

## Overview :mag:
This directory contains a sample dataset with the sort of data we encounter in our daily work at Filament.  
Your objective is to demonstrate end-to-end ability in preparing, modeling, and serving an ML system using this data.  

This task will focus on:
- Data preprocessing (handling noise, advanced text normalization)  
- Model development (vectorization or fine-tuning a transformer; building your own model is a plus)  
- Model evaluation & improvement (tuning, autoML, or other advanced methods)  
- Model deployment (wrapping the trained model in a performant service, e.g., FastAPI or LitServe)  

We recommend spending **no more than 4 hours** on this assignment. We don't expect you to hand in a finished product.
It's important that you document your thought process.

---

## Task Scope :book:

### 1️⃣ Data Preparation
- Explore the dataset and highlight any issues or challenges.  
- Apply meaningful preprocessing beyond basic cleaning.  

### 2️⃣ Modeling
- Either vectorize and train a model, or fine-tune a transformer.  
- Justify your approach and framework choice.  

### 3️⃣ Evaluation & Tuning
- Define suitable metrics.  
- Show at least one attempt to improve performance.  

### 4️⃣ Deployment
- Expose the model through an API endpoint `/predict` that accepts raw text and returns predictions.  

---

## What We Are Evaluating :microscope:
- **Advanced Data Preprocessing**: Ability to identify and address text-specific challenges, going beyond boilerplate cleaning.  
- **Modeling Expertise**: Depth of knowledge in ML implementations for training and/or fine-tuning, and clarity of rationale.  
- **Deployment Practices**: Understanding and use of best practices for deploying models in hybrid architectures (local + cloud-based).  
- **Systems Thinking (Optional/Bonus)**:  
If you have experience with agentic solutions or evaluation systems (e.g., pipeline orchestration, LLM-based evaluators, explainability frameworks), demonstrate how you would extend your solution with:

An evaluation component that goes beyond standard metrics (e.g., automatic scoring, human-in-the-loop feedback, adversarial testing).

OR an agentic approach (e.g., chaining multiple model calls, routing queries to specialized sub-models, or adding a retriever component for grounding).

This can be presented as code, a small prototype, or simply a diagram/explanation in your README. 

---

## Submission Guidelines :inbox_tray:
- Submit a `.zip` or GitHub repo with:  
  - All required code.  
  - A `README.md` explaining setup steps and your approach.  
- Briefly note what you would extend or improve if you had more time.  

---

## Recommended Libraries (but not limited to) :toolbox:
- **Data manipulation**: `pandas`, `numpy`  
- **NLP preprocessing**: `spaCy`, `NLTK`, `transformers` (Hugging Face)  
- **Modeling**: `PyTorch Lightning` or `TensorFlow`  
- **Visualization**: `matplotlib`, `seaborn`  
- **Deployment**: `FastAPI`, `LitServe`  

---

⌛ **Reminder:** Please limit your time on this assessment to **4 hours maximum**.  
We value depth and clarity over completeness of all sections.