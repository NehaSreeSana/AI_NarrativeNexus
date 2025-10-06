import os
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from .text_procesing import X_train, y_train, X_test, y_test


BASE_DIR = Path(__file__).resolve().parents[2]  # .../WORK5
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Vectorize entire dataset
vectorizer = TfidfVectorizer(max_features=3000, sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    max_features="sqrt",
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
rf.fit(X_train_tfidf, y_train)
y_pred_rf = rf.predict(X_test_tfidf)


# Evaluation
print(" Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# Confusion Matrix → save to WORK5 root
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
cm_path = BASE_DIR / "rf_confusion_matrix.png"
plt.savefig(cm_path, bbox_inches="tight")
plt.close()
print(f"Saved confusion matrix at: {cm_path}")


# Save artifacts under models/
rf_path = MODELS_DIR / "random_forest_model.pkl"
vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
joblib.dump(rf, rf_path)
joblib.dump(vectorizer, vec_path)
print(f"Saved model at: {rf_path}")
print(f"Saved vectorizer at: {vec_path}")


