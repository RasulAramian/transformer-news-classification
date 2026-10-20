from pathlib import Path

import joblib
import numpy as np
import torch

from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data_loader import load_ag_news


# ==================================================
# Project paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOGISTIC_REGRESSION_DIR = (
    PROJECT_ROOT / "models" / "logistic_regression"
)

# New training outputs: keep existing model untouched.
DISTILBERT_CHECKPOINT_DIR = (
    PROJECT_ROOT / "outputs" / "distilbert-ag-news-checkpoints"
)

DISTILBERT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "distilbert-ag-news"
)


# ==================================================
# 1. Logistic Regression training
# ==================================================

def train_logistic_regression():
    """Train and save the TF-IDF Logistic Regression baseline."""

    dataset = load_ag_news()

    texts = dataset["train"]["text"]
    labels = dataset["train"]["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.1,
        random_state=42,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    accuracy = model.score(X_val_tfidf, y_val)
    print(f"Logistic Regression validation accuracy: {accuracy:.4f}")

    LOGISTIC_REGRESSION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        LOGISTIC_REGRESSION_DIR / "logistic_regression.joblib",
    )

    joblib.dump(
        vectorizer,
        LOGISTIC_REGRESSION_DIR / "tfidf_vectorizer.joblib",
    )

    print(f"Logistic Regression saved to: {LOGISTIC_REGRESSION_DIR}")

    return model, vectorizer


# ==================================================
# 2. DistilBERT training
# ==================================================

def train_distilbert():
    """Fine-tune DistilBERT and save it to a separate output directory."""

    # Load AG News.
    dataset = load_ag_news()

    texts = dataset["train"]["text"]
    labels = dataset["train"]["label"]

    # Create a stratified training/validation split.
    train_texts, val_texts, train_labels, val_labels = (
        train_test_split(
            texts,
            labels,
            test_size=0.1,
            random_state=42,
            stratify=labels,
        )
    )

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Load the pretrained tokenizer and model.
    model_name = "distilbert/distilbert-base-uncased"
    max_length = 128

    id2label = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech",
    }

    label2id = {
        label: label_id
        for label_id, label in id2label.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
        id2label=id2label,
        label2id=label2id,
    )

    # Convert text into tokens.
    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    # Convert the data into Hugging Face Dataset objects.
    train_dataset = Dataset.from_dict({
        "text": list(train_texts),
        "label": list(train_labels),
    })

    val_dataset = Dataset.from_dict({
        "text": list(val_texts),
        "label": list(val_labels),
    })

    # Tokenize both datasets.
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
    )

    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
    )

    # The original text is no longer needed after tokenization.
    train_dataset = train_dataset.remove_columns(["text"])
    val_dataset = val_dataset.remove_columns(["text"])

    # Dynamically pad each batch.
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    # Calculate evaluation metrics.
    def compute_metrics(eval_pred):
        logits, true_labels = eval_pred

        predictions = np.argmax(logits, axis=-1)

        return {
            "accuracy": accuracy_score(
                true_labels,
                predictions,
            ),
            "macro_f1": f1_score(
                true_labels,
                predictions,
                average="macro",
            ),
        }

    # Configure the training process.
    training_args = TrainingArguments(
        output_dir=str(DISTILBERT_CHECKPOINT_DIR),
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=1,
        seed=42,
    )

    # Create the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Fine-tune DistilBERT.
    print("Starting DistilBERT training...")
    trainer.train()

    # Evaluate the best model on the validation dataset.
    validation_results = trainer.evaluate()

    print("\nDistilBERT validation results:")
    print(validation_results)

    # Save the trained model separately from the existing model.
    DISTILBERT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(str(DISTILBERT_OUTPUT_DIR))
    tokenizer.save_pretrained(str(DISTILBERT_OUTPUT_DIR))

    print(f"\nDistilBERT saved to: {DISTILBERT_OUTPUT_DIR}")

    return trainer
        

if __name__ == "__main__":
    train_logistic_regression()
