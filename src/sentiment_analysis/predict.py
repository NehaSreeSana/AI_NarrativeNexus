from pathlib import Path

import joblib
from .text_procesing import clean_text


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"


# Load saved RF + TF-IDF
rf_loaded = joblib.load(MODELS_DIR / "random_forest_model.pkl")
vectorizer_loaded = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")


def predict_rf(text: str) -> str:
    text_clean = clean_text(text)
    vec = vectorizer_loaded.transform([text_clean])
    pred = rf_loaded.predict(vec)[0]
    return "Positive" if pred == 1 else "Negative"


try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    lstm_loaded = load_model(MODELS_DIR / "lstm_model.h5")
    tokenizer_loaded = joblib.load(MODELS_DIR / "lstm_tokenizer.pkl")
    LSTM_AVAILABLE = True
except Exception as e:
    print("LSTM not available for predictions. Details:", e)
    LSTM_AVAILABLE = False


def predict_lstm(text: str) -> str:
    if not LSTM_AVAILABLE:
        return "LSTM unavailable"
    text_clean = clean_text(text)
    seq = tokenizer_loaded.texts_to_sequences([text_clean])
    padded = pad_sequences(seq, maxlen=200)
    pred = (lstm_loaded.predict(padded) > 0.5).astype("int32")[0][0]
    return "Positive" if pred == 1 else "Negative"


if __name__ == "__main__":
    sample_texts = [
        "This product is absolutely fantastic, I loved it!",
        "Worst purchase ever, total waste of money.",
        "It was okay, nothing special but not bad either.",
    ]

    for txt in sample_texts:
        print(f"\nINPUT: {txt}")
        print("Random Forest →", predict_rf(txt))
        print("LSTM          →", predict_lstm(txt))


