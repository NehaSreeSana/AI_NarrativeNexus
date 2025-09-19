import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Initialize NLTK components
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def nlp_preprocess(text):
    """Tokenize, clean, remove stopwords, lemmatize"""
    if not isinstance(text, str):
        return ""
    
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def preprocess_series(X):
    """Apply nlp_preprocess to a list/Series"""
    return [nlp_preprocess(t) for t in X]

pipeline = joblib.load("models/topic_classifier.pkl")

examples = [
    "The new graphics card from NVIDIA has amazing performance for 3D rendering.",
    "God does not exist, and religion is just a human creation.",
    "The new Windows update has caused issues with drivers.",
    "NASA discovered water on Mars, confirming planetary research findings."
]

predictions = pipeline.predict(examples)

for text, label in zip(examples, predictions):
    print(f"\n Text: {text}\n Predicted Category: {label}")
print("\n Predictions complete.")