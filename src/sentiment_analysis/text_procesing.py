import re
import os
import pandas as pd
import nltk

# Try to ensure required NLTK resources are available without spamming downloads
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _ = stopwords.words("english")
except Exception:  # LookupError or first-time import
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Basic cleaning: lowercasing, removing special chars, stopwords, lemmatization"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)


# Full dataset paths (Windows raw strings)
train_path = r"C:\Users\HP\Downloads\WORK5\req_data\amazon_rev\amazon_reviews_train.csv"
test_path = r"C:\Users\HP\Downloads\WORK5\req_data\amazon_rev\amazon_reviews_test.csv"


# Optional fast-mode controls via environment variables
max_train = os.getenv("SA_MAX_TRAIN_ROWS")
max_test = os.getenv("SA_MAX_TEST_ROWS")
sample_frac = os.getenv("SA_SAMPLE_FRAC")

# Load CSVs with optional row limits
train_df = pd.read_csv(train_path, nrows=int(max_train) if max_train else None)
test_df = pd.read_csv(test_path, nrows=int(max_test) if max_test else None)

# Optional fractional sampling for quick experiments
if sample_frac:
    try:
        frac = float(sample_frac)
        if 0 < frac < 1:
            train_df = train_df.sample(frac=frac, random_state=42).reset_index(drop=True)
            test_df = test_df.sample(frac=frac, random_state=42).reset_index(drop=True)
    except ValueError:
        pass


# Build text field from title + content then clean
train_df["text"] = (train_df["title"].astype(str) + " " + train_df["content"].astype(str)).apply(clean_text)
test_df["text"] = (test_df["title"].astype(str) + " " + test_df["content"].astype(str)).apply(clean_text)


# Expose train/test splits
X_train = train_df["text"]
y_train = train_df["label"]
X_test = test_df["text"]
y_test = test_df["label"]


print("Train size:", len(X_train), " Test size:", len(X_test))


