from pathlib import Path

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

LOGISTIC_REGRESSION_DIR = MODELS_DIR / "logistic_regression"
DISTILBERT_MODEL_ID = "RasulAramian/distilbert-ag-news"


def load_logistic_regression():
    """Load the trained Logistic Regression model and TF-IDF vectorizer."""

    model_path = LOGISTIC_REGRESSION_DIR / "logistic_regression.joblib"
    vectorizer_path = LOGISTIC_REGRESSION_DIR / "tfidf_vectorizer.joblib"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


def load_distilbert():
    """Load the fine-tuned DistilBERT model and tokenizer from Hugging Face."""

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_ID)

    model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL_ID
    )

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, device
