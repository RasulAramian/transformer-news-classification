from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from src.data_loader import load_ag_news
from src.model import load_logistic_regression


def evaluate_logistic_regression():
    """Evaluate the saved Logistic Regression model on validation data."""

    dataset = load_ag_news()

    texts = dataset["train"]["text"]
    labels = dataset["train"]["label"]

    _, X_val, _, y_val = train_test_split(
        texts,
        labels,
        test_size=0.1,
        random_state=42,
        stratify=labels,
    )

    model, vectorizer = load_logistic_regression()

    X_val_tfidf = vectorizer.transform(X_val)
    predictions = model.predict(X_val_tfidf)

    print(f"Accuracy: {accuracy_score(y_val, predictions):.4f}")
    print(f"Macro F1: {f1_score(y_val, predictions, average='macro'):.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_val,
            predictions,
            target_names=["World", "Sports", "Business", "Sci/Tech"],
        )
    )


if __name__ == "__main__":
    evaluate_logistic_regression()
