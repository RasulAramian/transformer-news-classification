# Transformer-Based News Classification

An end-to-end Natural Language Processing (NLP) project that compares a traditional machine learning baseline with a fine-tuned Transformer model for multi-class news classification.

The project uses the AG News dataset to classify news articles into four categories: World, Sports, Business, and Sci/Tech.

It covers data preparation, model training, evaluation, inference, API development, containerization, and automated testing.

## Project Highlights

- Implemented a TF-IDF + Logistic Regression baseline.
- Fine-tuned DistilBERT using PyTorch and Hugging Face Transformers.
- Evaluated both models on the same 7,600 test articles.
- Achieved 94.86% test accuracy with DistilBERT.
- Improved classification accuracy by 3.23 percentage points over the baseline.
- Published the fine-tuned Transformer model on Hugging Face.
- Developed a FastAPI inference service.
- Containerized the API using Docker.
- Added automated API and model tests with pytest.

## Dataset

This project uses the AG News dataset, available through Hugging Face.

Dataset: https://huggingface.co/datasets/fancyzhx/ag_news

The dataset contains 120,000 training examples and 7,600 test examples.

The original training split was divided into:

| Split | Samples |
|---|---:|
| Training | 108,000 |
| Validation | 12,000 |
| Test | 7,600 |

A stratified 90/10 training-validation split was created using a random seed of 42.

The official test set was kept separate for final evaluation.

### Classification Categories

| Label | Category |
|---|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

## Models

### 1. TF-IDF + Logistic Regression

The baseline combines TF-IDF text vectorization with a Logistic Regression classifier.

This approach provides a computationally efficient reference model for evaluating the benefits of Transformer-based classification.

The trained classifier and TF-IDF vectorizer are stored in:

`models/logistic_regression/`

### 2. Fine-Tuned DistilBERT

The Transformer model is based on:

`distilbert/distilbert-base-uncased`

DistilBERT was fine-tuned for four-class news classification using the Hugging Face Trainer API.

### Training Configuration

| Parameter | Value |
|---|---|
| Base model | DistilBERT Base Uncased |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Training batch size | 16 |
| Evaluation batch size | 32 |
| Maximum sequence length | 128 |
| Weight decay | 0.01 |
| Random seed | 42 |
| Evaluation strategy | Every epoch |
| Checkpoint strategy | Every epoch |
| Best model selection | Validation accuracy |
| Mixed precision | FP16 |
| Training hardware | NVIDIA Tesla T4 |

Dynamic padding was used to reduce unnecessary computation.

The model was trained using GPU acceleration in Google Colab.

## Model Evaluation

Both models were evaluated on the same official AG News test set containing 7,600 articles.

### Performance Comparison

| Metric | Logistic Regression | DistilBERT |
|---|---:|---:|
| Accuracy | 91.63% | **94.86%** |
| Macro Precision | 91.61% | **94.88%** |
| Macro Recall | 91.63% | **94.86%** |
| Macro F1-score | 91.61% | **94.86%** |
| Inference time | 0.91 s | 27.40 s |
| Correct predictions | 6,964 | **7,209** |
| Incorrect predictions | 636 | **391** |

Inference times represent full-test-set measurements in the notebook environment, including text preprocessing or tokenization.

These timings are hardware-dependent and should not be interpreted as production API latency benchmarks.

### Key Findings

DistilBERT achieved a 3.23 percentage-point improvement in classification accuracy compared with Logistic Regression.

The number of incorrect predictions decreased from 636 to 391, representing approximately a 38.5% reduction in classification errors.

Logistic Regression demonstrated substantially faster inference in the measured experiment.

These results highlight the trade-off between predictive performance and computational cost.

## Hugging Face Model

The fine-tuned DistilBERT model is available on Hugging Face:

**[RasulAramian/distilbert-ag-news](https://huggingface.co/RasulAramian/distilbert-ag-news)**

The model can be loaded directly using Hugging Face Transformers.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "RasulAramian/distilbert-ag-news"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
```

## API and Deployment

The project includes a FastAPI application for model inference.

The API provides:

- `/health` — Application health and model-loading status.
- `/predict` — News classification predictions.

The application can be containerized using Docker.

### Local Development

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open the interactive API documentation:

http://localhost:8000/docs

### Docker

Build the Docker image:

```bash
docker build -t transformer-news-classification-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 transformer-news-classification-api
```

Check the health endpoint:

```bash
curl http://localhost:8000/health
```

The application requires access to the model artifacts used for inference.

## Testing

The project includes automated tests for API functionality and model predictions.

Run the test suite using:

```bash
pytest -v
```

## Technologies

**Machine Learning**
- Python
- scikit-learn
- TF-IDF
- Logistic Regression

**Deep Learning and NLP**
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- DistilBERT

**API and Deployment**
- FastAPI
- Pydantic
- Uvicorn
- Docker

**Development and Testing**
- Git
- GitHub
- pytest
- Jupyter Notebook

## Project Workflow

1. Load and explore the AG News dataset.
2. Prepare the training, validation, and test data.
3. Train the Logistic Regression baseline.
4. Fine-tune DistilBERT for news classification.
5. Evaluate both models using common metrics.
6. Analyze classification errors and confusion matrices.
7. Publish the Transformer model on Hugging Face.
8. Integrate trained models into an inference API.
9. Containerize and test the application.

## Conclusion

This project demonstrates the development and evaluation of two approaches to multi-class text classification.

DistilBERT achieved higher predictive performance, while Logistic Regression offered a simpler and faster inference solution.

The project also demonstrates practical experience with Transformer fine-tuning, model evaluation, artifact management, API development, Docker, and automated testing.

## Author

**Rasul Aramian**

GitHub: https://github.com/RasulAramian

Hugging Face: https://huggingface.co/RasulAramian
