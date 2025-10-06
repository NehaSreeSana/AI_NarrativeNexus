from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from .text_procesing import X_test, y_test


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"


def load_predictions():
    # Lazy imports to avoid heavy deps if only tabulating
    import joblib
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from .text_procesing import clean_text

    # RF
    rf_loaded = joblib.load(MODELS_DIR / "random_forest_model.pkl")
    vectorizer_loaded = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    X_test_vec = vectorizer_loaded.transform(X_test)
    y_pred_rf = rf_loaded.predict(X_test_vec)

    # LSTM (optional)
    try:
        lstm_loaded = load_model(MODELS_DIR / "lstm_model.h5")
        tokenizer_loaded = joblib.load(MODELS_DIR / "lstm_tokenizer.pkl")
        X_test_seq = pad_sequences(tokenizer_loaded.texts_to_sequences(X_test), maxlen=200)
        y_pred_lstm = (lstm_loaded.predict(X_test_seq) > 0.5).astype("int32").flatten()
    except Exception as e:
        print("LSTM model not available; skipping LSTM evaluation. Details:", e)
        y_pred_lstm = None

    return y_pred_rf, y_pred_lstm


if __name__ == "__main__":
    y_pred_rf, y_pred_lstm = load_predictions()

    results = {"Model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1-Score": []}

    # RF row
    results["Model"].append("Random Forest")
    results["Accuracy"].append(accuracy_score(y_test, y_pred_rf))
    results["Precision"].append(precision_score(y_test, y_pred_rf))
    results["Recall"].append(recall_score(y_test, y_pred_rf))
    results["F1-Score"].append(f1_score(y_test, y_pred_rf))

    # LSTM row only if available
    if y_pred_lstm is not None:
        results["Model"].append("LSTM")
        results["Accuracy"].append(accuracy_score(y_test, y_pred_lstm))
        results["Precision"].append(precision_score(y_test, y_pred_lstm))
        results["Recall"].append(recall_score(y_test, y_pred_lstm))
        results["F1-Score"].append(f1_score(y_test, y_pred_lstm))

    df_results = pd.DataFrame(results)
    print(" Model Performance Summary")
    print(df_results.to_string(index=False))


