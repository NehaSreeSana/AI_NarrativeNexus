from pathlib import Path

import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, accuracy_score

from .text_procesing import X_train, y_train, X_test, y_test


# Ensure required NLTK tokenizers are available for sentence tokenization
for resource in ['punkt', 'punkt_tab']:
    try:
        # punkt lives under tokenizers/punkt; punkt_tab under tokenizers/punkt_tab
        subdir = 'tokenizers/punkt' if resource == 'punkt' else 'tokenizers/punkt_tab'
        nltk.data.find(subdir)
    except LookupError:
        nltk.download(resource, quiet=True)


analyzer = SentimentIntensityAnalyzer()


def get_sentiment_sentence_level(text: str) -> int:
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except LookupError:
        # Try to fetch missing resources at runtime
        for resource in ['punkt', 'punkt_tab']:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
        except Exception:
            # Final fallback: simple regex split
            import re
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]

    if not sentences:
        return 1
    scores = [analyzer.polarity_scores(sentence)["compound"] for sentence in sentences]
    average_compound = sum(scores) / len(scores)
    return 1 if average_compound >= 0 else 0


if __name__ == "__main__":
    # Predict on train and test
    y_pred_train = [get_sentiment_sentence_level(text) for text in X_train]
    y_pred_test = [get_sentiment_sentence_level(text) for text in X_test]

    # Evaluate
    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
    print("\nTest Classification Report:\n")
    print(classification_report(y_test, y_pred_test))


