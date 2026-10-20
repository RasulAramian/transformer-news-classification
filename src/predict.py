import torch

from src.model import load_distilbert, load_logistic_regression
from src.preprocessing import normalize_text


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


class NewsPredictor:
    """Predict news categories using the saved classification models."""

    def __init__(self):
        # Load both trained models once per predictor instance
        self.lr_model, self.vectorizer = load_logistic_regression()
        self.bert_model, self.tokenizer, self.device = load_distilbert()

    def predict_with_logistic_regression(self, text):
        """Predict a news category using TF-IDF and Logistic Regression."""

        # Validate and normalize input text
        text = normalize_text(text)

        # Convert text into TF-IDF features
        features = self.vectorizer.transform([text])

        # Predict category and probabilities
        prediction = int(self.lr_model.predict(features)[0])
        probabilities = self.lr_model.predict_proba(features)[0]

        return {
            "model": "logistic_regression",
            "label": LABEL_NAMES[prediction],
            "confidence": float(probabilities[prediction]),
        }

    def predict_with_distilbert(self, text):
        """Predict a news category using DistilBERT."""

        # Validate and normalize input text
        text = normalize_text(text)

        # Tokenize the input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)

        # Run inference without calculating gradients
        with torch.inference_mode():
            outputs = self.bert_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        # Select the category with the highest probability
        prediction = int(probabilities.argmax().item())

        return {
            "model": "distilbert",
            "label": LABEL_NAMES[prediction],
            "confidence": float(probabilities[prediction].item()),
        }
