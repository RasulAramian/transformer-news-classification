from src.predict import NewsPredictor


def test_models_predict_valid_categories():
    # Load both trained models
    predictor = NewsPredictor()

    # Sample news text
    text = (
        "The stock market rose after the government "
        "announced a new economic policy."
    )

    # Get predictions from both models
    lr_result = predictor.predict_with_logistic_regression(text)
    bert_result = predictor.predict_with_distilbert(text)

    # Valid news categories
    valid_labels = {"World", "Sports", "Business", "Sci/Tech"}

    # Check Logistic Regression output
    assert lr_result["model"] == "logistic_regression"
    assert lr_result["label"] in valid_labels
    assert 0.0 <= lr_result["confidence"] <= 1.0

    # Check DistilBERT output
    assert bert_result["model"] == "distilbert"
    assert bert_result["label"] in valid_labels
    assert 0.0 <= bert_result["confidence"] <= 1.0
