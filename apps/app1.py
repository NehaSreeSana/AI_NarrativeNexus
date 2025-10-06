
import os
from pathlib import Path

import streamlit as st
import joblib

# Optional imports guarded for environments without TensorFlow or VADER
try:
    from tensorflow.keras.models import load_model as tf_load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

try:
    import nltk
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False


# Local imports
from src.sentiment_analysis.text_procesing import clean_text


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


@st.cache_resource(show_spinner=False)
def load_rf():
    rf_path = MODELS_DIR / "random_forest_model.pkl"
    vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    if not rf_path.exists() or not vec_path.exists():
        return None, None
    rf = joblib.load(rf_path)
    vectorizer = joblib.load(vec_path)
    return rf, vectorizer


@st.cache_resource(show_spinner=False)
def load_lstm():
    if not TENSORFLOW_AVAILABLE:
        return None, None
    lstm_path = MODELS_DIR / "lstm_model.h5"
    tok_path = MODELS_DIR / "lstm_tokenizer.pkl"
    if not lstm_path.exists() or not tok_path.exists():
        return None, None
    model = tf_load_model(lstm_path)
    tokenizer = joblib.load(tok_path)
    return model, tokenizer


@st.cache_resource(show_spinner=False)
def load_vader():
    if not VADER_AVAILABLE:
        return None
    # Ensure required tokenizers exist for sentence-level analysis
    try:
        for resource in ["punkt", "punkt_tab"]:
            try:
                subdir = "tokenizers/punkt" if resource == "punkt" else "tokenizers/punkt_tab"
                nltk.data.find(subdir)
            except LookupError:
                nltk.download(resource, quiet=True)
    except Exception:
        pass
    return SentimentIntensityAnalyzer() if VADER_AVAILABLE else None


def predict_with_rf(text: str, rf, vectorizer) -> str:
    text_clean = clean_text(text)
    vec = vectorizer.transform([text_clean])
    pred = rf.predict(vec)[0]
    return "Positive" if int(pred) == 1 else "Negative"


def predict_with_lstm(text: str, model, tokenizer) -> str:
    text_clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([text_clean])
    # Match training input_length if available; default to 200
    try:
        input_len = int(model.input_shape[1])
    except Exception:
        input_len = 200
    padded = pad_sequences(seq, maxlen=input_len)
    pred = (model.predict(padded) > 0.5).astype("int32")[0][0]
    return "Positive" if int(pred) == 1 else "Negative"


def predict_with_vader(text: str, analyzer) -> str:
    # Sentence-level averaging; fallback to single compound
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [text]
    if not sentences:
        sentences = [text]
    scores = [analyzer.polarity_scores(s)["compound"] for s in sentences]
    average_compound = sum(scores) / len(scores)
    return "Positive" if average_compound >= 0 else "Negative"


st.set_page_config(page_title="Sentiment Predictor", page_icon="💬", layout="centered")
st.title("Sentiment Prediction App")
st.write("Enter a sentence, choose a model, and get Positive/Negative prediction.")

rf_model, rf_vectorizer = load_rf()
lstm_model, lstm_tokenizer = load_lstm()
vader_analyzer = load_vader()

available_models = []
if rf_model and rf_vectorizer:
    available_models.append("Random Forest")
if lstm_model and lstm_tokenizer:
    available_models.append("LSTM")
if vader_analyzer:
    available_models.append("VADER")

if not available_models:
    st.error("No models available. Train or place artifacts in the 'models' folder, or install VADER.")
else:
    model_choice = st.selectbox("Choose a model", options=available_models, index=0)
    text = st.text_area("Input sentence", height=120, placeholder="Type a sentence...")
    if st.button("Predict"):
        if not text.strip():
            st.warning("Please enter a sentence.")
        else:
            with st.spinner("Predicting..."):
                try:
                    if model_choice == "Random Forest":
                        pred = predict_with_rf(text, rf_model, rf_vectorizer)
                    elif model_choice == "LSTM":
                        pred = predict_with_lstm(text, lstm_model, lstm_tokenizer)
                    else:
                        pred = predict_with_vader(text, vader_analyzer)
                    st.success(f"Prediction: {pred}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

st.caption("Models are loaded from the 'models' folder. VADER requires 'vaderSentiment'.")

import sys
import argparse
from pathlib import Path

import joblib

from src.sentiment_analysis.text_procesing import clean_text


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def predict_with_rf(text: str) -> str:
    model_path = MODELS_DIR / "random_forest_model.pkl"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError("Random Forest artifacts not found in models/. Train the RF model first.")
    rf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    text_clean = clean_text(text)
    vec = vectorizer.transform([text_clean])
    pred = rf.predict(vec)[0]
    return "Positive" if int(pred) == 1 else "Negative"


def predict_with_lstm(text: str) -> str:
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except Exception as e:
        raise RuntimeError(f"TensorFlow is not available: {e}")

    model_path = MODELS_DIR / "lstm_model.h5"
    tokenizer_path = MODELS_DIR / "lstm_tokenizer.pkl"
    if not model_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError("LSTM artifacts not found in models/. Train the LSTM model first.")
    lstm = load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)

    text_clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([text_clean])
    padded = pad_sequences(seq, maxlen=200)
    pred = (lstm.predict(padded) > 0.5).astype("int32")[0][0]
    return "Positive" if int(pred) == 1 else "Negative"


def predict_with_vader(text: str) -> str:
    import nltk
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception as e:
        raise RuntimeError(f"vaderSentiment is not available: {e}")

    # Ensure tokenizers are present
    for resource in ['punkt', 'punkt_tab']:
        try:
            subdir = 'tokenizers/punkt' if resource == 'punkt' else 'tokenizers/punkt_tab'
            nltk.data.find(subdir)
        except LookupError:
            nltk.download(resource, quiet=True)

    analyzer = SentimentIntensityAnalyzer()

    # Sentence-level aggregation
    def tokenize_sentences(t: str):
        try:
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(t)
        except Exception:
            import re
            return [s for s in re.split(r"(?<=[.!?])\s+", t) if s]

    sentences = tokenize_sentences(text)
    if not sentences:
        return "Positive"
    scores = [analyzer.polarity_scores(s)["compound"] for s in sentences]
    avg = sum(scores) / len(scores)
    return "Positive" if avg >= 0 else "Negative"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Predict sentiment (Positive/Negative) using RF, LSTM, or VADER.")
    parser.add_argument("--model", choices=["rf", "lstm", "vader"], required=True, help="Model to use for prediction")
    parser.add_argument("--text", type=str, help="Input text to classify. If omitted, you will be prompted.")
    args = parser.parse_args(argv)

    text = args.text
    if not text:
        try:
            text = input("Enter text: ").strip()
        except KeyboardInterrupt:
            return 1

    if not text:
        print("No text provided.")
        return 1

    if args.model == "rf":
        label = predict_with_rf(text)
    elif args.model == "lstm":
        label = predict_with_lstm(text)
    else:
        label = predict_with_vader(text)

    print(label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


