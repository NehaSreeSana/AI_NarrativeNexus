from pathlib import Path

import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
except Exception as e:
    print("TensorFlow/Keras not available. Skipping LSTM training. Details:", e)
    sys.exit(0)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from .text_procesing import X_train, y_train, X_test, y_test


BASE_DIR = Path(__file__).resolve().parents[2]  # .../WORK5
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Tokenization on full dataset
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=120)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=120)


# Build model
model_lstm = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=120),
    LSTM(64, dropout=0.2),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model_lstm.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# Train (keep epochs small for initial run)
callbacks = [
    EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1)
]
history = model_lstm.fit(
    X_train_seq, np.array(y_train),
    epochs=3, batch_size=256,
    validation_data=(X_test_seq, np.array(y_test)),
    callbacks=callbacks,
    verbose=1
)


# Evaluate
y_pred_lstm = (model_lstm.predict(X_test_seq, batch_size=1024) > 0.5).astype("int32").flatten()
print(" LSTM Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lstm))
print(classification_report(y_test, y_pred_lstm))


# Confusion Matrix → save to WORK5 root
cm = confusion_matrix(y_test, y_pred_lstm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("LSTM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
cm_path = BASE_DIR / "lstm_confusion_matrix.png"
plt.savefig(cm_path, bbox_inches="tight")
plt.close()
print(f"Saved confusion matrix at: {cm_path}")


# Save artifacts under models/
lstm_path = MODELS_DIR / "lstm_model.h5"
tokenizer_path = MODELS_DIR / "lstm_tokenizer.pkl"
model_lstm.save(lstm_path)
joblib.dump(tokenizer, tokenizer_path)
print(f"Saved model at: {lstm_path}")
print(f"Saved tokenizer at: {tokenizer_path}")


