def normalize_text(text: str) -> str:
    """Remove unnecessary whitespace around a news article."""

    if not isinstance(text, str):
        raise TypeError("News text must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("News text cannot be empty.")

    return text
