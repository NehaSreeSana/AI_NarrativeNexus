import streamlit as st
import joblib
import os
import sys
from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load .env if present
load_dotenv()

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

# Mapping from raw 20 Newsgroups labels to display-friendly names
CATEGORY_DISPLAY_NAMES = {
    "alt.atheism": "atheism",
    "comp.graphics": "computer graphics",
    "comp.os.ms-windows.misc": "windows os",
    "comp.sys.ibm.pc.hardware": "system pc hardware",
    "comp.sys.mac.hardware": "mac hardware",
    "comp.windows.x": "computer windows",
    "misc.forsale": "forsale",
    "rec.autos": "autos",
    "rec.motorcycles": "motorcycles",
    "rec.sport.baseball": "baseball sport",
    "rec.sport.hockey": "hockey sport",
    "sci.crypt": "cryptography",
    "sci.electronics": "electronics",
    "sci.med": "medical",
    "sci.space": "space",
    "soc.religion.christian": "religion christian",
    "talk.politics.guns": "politics guns",
    "talk.politics.mideast": "politics mideast",
    "talk.politics.misc": "politics",
    "talk.religion.misc": "religion",
}

# Page configuration
st.set_page_config(
    page_title="Text Classification Predictor",
    page_icon="📝",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .category-text {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 0.75rem;
    }
    .stButton > button {
        width: 20%;
        font-size: 1.1rem;
        padding: 0.75rem;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained model"""
    model_path = "models/topic_classifier.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please train the model first by running the training script.")
        return None
    try:
        # Use joblib to load the model
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {str(e)}. Please retrain the model first.")
        return None

def extract_category_name(full_category):
    """Return display-friendly name for a raw category label."""
    return CATEGORY_DISPLAY_NAMES.get(full_category, full_category.split('.')[-1])

def predict_category(text, pipeline):
    """Predict the category for the given text"""
    try:
        # Preprocess the text
        processed_text = preprocess_series([text])
        
        # Make prediction
        prediction = pipeline.predict(processed_text)[0]
        
        # Extract category name
        category_name = extract_category_name(prediction)
        
        return prediction, category_name
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

def predict_with_confidence(text, pipeline):
    """Predict category and try to compute confidence if the model supports it."""
    label, name = predict_category(text, pipeline)
    confidence = None
    try:
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(preprocess_series([text]))[0]
            confidence = float(max(probs)) if probs is not None else None
        elif hasattr(pipeline, "decision_function"):
            # Approximate confidence from decision function by normalizing margins
            margins = pipeline.decision_function(preprocess_series([text]))
            # Handle binary vs multi-class shapes
            import numpy as _np
            arr = _np.atleast_1d(margins)
            if arr.ndim == 1:
                # Binary: map margin -> pseudo-probability via logistic
                confidence = float(1 / (1 + _np.exp(-abs(arr[0]))))
            else:
                # Multi-class: softmax over margins
                exp_m = _np.exp(arr - _np.max(arr))
                softmax = exp_m / _np.sum(exp_m)
                confidence = float(_np.max(softmax))
    except Exception:
        confidence = None
    return label, name, confidence

def fetch_text_from_url(url, meta_only=False):
    """Fetch and extract main text content from Reddit, news pages, or Hugging Face cards.
    This is a lightweight heuristic extractor to avoid heavy dependencies.
    """
    try:
        # Compose headers with optional API tokens
        headers = {"User-Agent": os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")}
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token and re.search(r"huggingface\.co", url):
            headers["Authorization"] = f"Bearer {hf_token}"
        reddit_token = os.getenv("REDDIT_TOKEN")
        if reddit_token and re.search(r"reddit\.com", url):
            headers["Authorization"] = f"Bearer {reddit_token}"

        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
    except Exception as e:
        st.error(f"Failed to fetch URL: {e}")
        return ""

    # Reddit handling is done in fetch_meta_summary for meta-only and below for full extraction

    # Hugging Face pages often have descriptions in meta tags
    if re.search(r"huggingface\.co", url):
        # Continue to generic extraction; meta tags included
        pass

    # Parse HTML
    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception:
        return ""

    # Remove script/style and hidden elements
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.extract()
    for el in soup.select('[aria-hidden="true"], [hidden], .sr-only, .visually-hidden'):
        el.extract()

    # Prefer meta tags if rich summaries exist
    meta_bits = []
    def get_meta(name: str, attr: str = "name"):
        m = soup.find("meta", attrs={attr: name})
        return m.get("content", "").strip() if m and m.get("content") else ""

    meta_bits.append(get_meta("og:title", attr="property"))
    meta_bits.append(get_meta("og:description", attr="property"))
    meta_bits.append(get_meta("twitter:title"))
    meta_bits.append(get_meta("twitter:description"))
    meta_bits.append(get_meta("description"))
    # Fallback to <title>
    if soup.title and soup.title.string:
        meta_bits.append(soup.title.string.strip())
    meta_text = " ".join([b for b in meta_bits if b])

    if meta_only:
        return meta_text

def fetch_meta_summary(url):
    # Reddit-specific: try public JSON API for posts to avoid generic site title
    if re.search(r"reddit\\.com", url):
        try:
            # Ensure .json endpoint
            clean = re.split(r"[#?]", url)[0].rstrip("/")
            if not clean.endswith(".json"):
                clean = clean + ".json"
            headers = {"User-Agent": os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")}
            r = requests.get(clean, headers=headers, timeout=20)
            r.raise_for_status()
            data = r.json()
            # Typical structure: [ { data: { children: [ { data: {...} } ] } } ]
            post = None
            if isinstance(data, list) and data:
                listing = data[0].get("data", {}) if isinstance(data[0], dict) else {}
                children = listing.get("children", []) if isinstance(listing, dict) else []
                if children and isinstance(children[0], dict):
                    post = children[0].get("data", {})
            if post:
                title = (post.get("title") or "").strip()
                selftext = (post.get("selftext") or post.get("body") or "").strip()
                # Some posts put text in media metadata; we keep it simple
                fields = {
                    "og:title": title,
                    "og:description": selftext[:500],
                    "twitter:title": title,
                    "twitter:description": selftext[:500],
                    "description": selftext[:500],
                    "title": title,
                }
                combined = " ".join([v for v in fields.values() if v])
                combined = re.sub(r"\s+", " ", combined).strip()
                if combined:
                    return combined, fields
        except Exception:
            # Fall back to generic meta parsing below
            pass

    """Return (combined_meta_text, meta_fields_dict) from page head without heavy scraping."""
    try:
        headers = {"User-Agent": os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
    except Exception:
        return "", {}
    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception:
        return "", {}

    def get_meta(soup_obj, name: str, attr: str = "name"):
        m = soup_obj.find("meta", attrs={attr: name})
        return m.get("content", "").strip() if m and m.get("content") else ""

    fields = {
        "og:title": get_meta(soup, "og:title", attr="property"),
        "og:description": get_meta(soup, "og:description", attr="property"),
        "twitter:title": get_meta(soup, "twitter:title"),
        "twitter:description": get_meta(soup, "twitter:description"),
        "description": get_meta(soup, "description"),
        "title": (soup.title.string.strip() if soup.title and soup.title.string else ""),
    }
    combined = " ".join([v for v in fields.values() if v])
    combined = re.sub(r"\s+", " ", combined).strip()

    # If meta is missing/weak or generic, supplement with headings and first paragraphs
    generic_titles = {
        "reddit - the heart of the internet",
        "access denied",
        "just a moment",
        "403 forbidden",
        "page not found",
        "sign in",
        "sign up",
    }
    is_generic = (fields.get("title", "").strip().lower() in generic_titles) or (len(combined.split()) < 5)
    if is_generic:
        # Extract prominent heading and first meaningful paragraphs
        try:
            # Remove scripts/styles quickly
            for tag in soup(["script", "style", "noscript", "template"]):
                tag.extract()
            h = None
            for tag in ("h1", "h2", "h3"):
                node = soup.find(tag)
                if node and node.get_text(strip=True):
                    h = node.get_text(" ", strip=True)
                    break
            paras = []
            for p in soup.find_all("p"):
                t = p.get_text(" ", strip=True)
                if len(t.split()) >= 8 and not re.search(r"cookie|privacy|terms|subscribe|advert", t, re.I):
                    paras.append(t)
                if len(paras) >= 2:
                    break
            if h:
                fields["heading"] = h
            if paras:
                fields["first_paragraphs"] = " \n\n".join(paras)
            supplement = " ".join([h or "", " ".join(paras)])
            supplement = re.sub(r"\s+", " ", supplement).strip()
            if supplement:
                combined = (combined + " " + supplement).strip() if combined else supplement
        except Exception:
            pass
    return combined, fields

    # Build candidate content blocks and score them
    def compute_link_density(node_text: str, node) -> float:
        link_text_len = sum(len(a.get_text(" ", strip=True)) for a in node.find_all("a"))
        total_len = max(len(node_text), 1)
        return min(link_text_len / total_len, 1.0)

    def score_block(node) -> float:
        text = node.get_text(" ", strip=True)
        words = len(text.split())
        if words < 40:
            return 0.0
        ld = compute_link_density(text, node)
        heading_bonus = 1.0
        # Bonus if container includes headings or article semantics
        if node.find(["h1", "h2", "h3"]) or node.name in ("article", "main", "section"):
            heading_bonus += 0.2
        # Penalize if many nav/sidebar cues
        attr_text = " ".join([node.get("role", ""), node.get("class", [""]).__repr__()]).lower()
        if any(k in attr_text for k in ("nav", "menu", "sidebar", "footer", "header")):
            heading_bonus -= 0.4
        return words * (1.0 - ld) * heading_bonus

    selector_candidates = [
        "article",
        "main",
        "div[itemprop='articleBody']",
        "section[role='main']",
        "div[class*='article']",
        "div[class*='content']",
        "div[class*='post']",
        "section[class*='content']",
    ]

    best_node = None
    best_score = 0.0
    for sel in selector_candidates:
        for node in soup.select(sel):
            s = score_block(node)
            if s > best_score:
                best_score = s
                best_node = node

    # If no good node found, consider all <p> density in body
    if best_node is None:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        paragraphs = [p for p in paragraphs if len(p.split()) >= 8]
        raw_text = " ".join(paragraphs)
    else:
        # Gather paragraph text within the best node
        paragraphs = [p.get_text(" ", strip=True) for p in best_node.find_all("p")]
        paragraphs = [p for p in paragraphs if len(p.split()) >= 5]
        raw_text = " ".join(paragraphs) or best_node.get_text(" ", strip=True)

    # Prepend meta summary if available and not already contained
    if meta_text and meta_text not in raw_text:
        raw_text = f"{meta_text} \n\n{raw_text}"

    def strip_boilerplate_and_numbers(text: str, page_url: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        lines = re.split(r"(?<=[\.!?])\s+|\n+", text)
        site_keywords_common = (
            "cookie", "privacy", "terms", "subscribe", "newsletter", "sign in", "sign up",
            "login", "register", "menu", "search", "advertisement", "advert", "share",
            "related articles", "related content", "back to top", "all rights reserved", "©",
            "copyright", "read more", "leave a comment", "view video"
        )
        reddit_keywords = (
            "reddit", "user agreement", "accessibility", "create your account",
            "continue with email", "continue with phone number", "public anyone can view",
            "to this community"
        )
        hf_keywords = ("hugging face", "model card", "dataset card")

        filtered = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            # extremely short
            if len(s.split()) < 3:
                continue
            low = s.lower()
            if any(k in low for k in site_keywords_common):
                continue
            if re.search(r"reddit\.com", page_url) and any(k in low for k in reddit_keywords):
                continue
            if re.search(r"huggingface\.co", page_url) and any(k in low for k in hf_keywords):
                continue
            # high digit/symbol ratio
            digits_ratio = sum(ch.isdigit() for ch in s) / max(len(s), 1)
            if digits_ratio > 0.35:
                continue
            # timestamps or numeric-only
            if re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", s):
                continue
            if re.fullmatch(r"[\d\W_]+", s):
                continue
            filtered.append(s)

        cleaned = " ".join(filtered)
        # Remove very large numbers and large grouped numbers
        cleaned = re.sub(r"\b\d{5,}\b", " ", cleaned)
        cleaned = re.sub(r"\b\d{1,3}(?:[,\.\s]\d{3}){2,}\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        
        # Sentence-level filtering: drop boilerplate sentences, keep the rest in original order
        sentences = re.split(r"(?<=[\.!?])\s+", cleaned)
        boilerplate_phrases = (
            "got a question", "this is the place", "members online", "view video",
            "continue with email", "continue with phone number", "create your account",
            "public anyone can view", "privacy policy", "user agreement", "wiki"
        )
        kept_sentences = []
        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            # drop very short sentences
            if len(s.split()) < 2:
                continue
            if any(p in s.lower() for p in boilerplate_phrases):
                continue
            # drop numeric-dense sentences
            digits_ratio = sum(ch.isdigit() for ch in s) / max(len(s), 1)
            if digits_ratio > 0.35:
                continue
            kept_sentences.append(s)
        if kept_sentences:
            cleaned = " ".join(kept_sentences)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    extracted = strip_boilerplate_and_numbers(raw_text, url)
    return extracted

def main():
    # Header
    st.markdown('<h1 class="main-header">📝 Text Classification Predictor</h1>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading the trained model..."):
        pipeline = load_model()
    
    if pipeline is None:
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Main interface
    tabs = st.tabs(["Text", "URL (Reddit/News/Hugging Face)"])

    with tabs[0]:
        st.markdown("### Enter a sentence to classify:")
        user_text = st.text_area(
            "Type your sentence here:",
            placeholder="Example: The new graphics card from NVIDIA has amazing performance for 3D rendering.",
            height=100,
            help="Enter any text and click 'Predict Category' to see which category it belongs to.",
            key="input_text"
        )
        predict_button = st.button("🔍 Predict Category", type="primary", use_container_width=True, key="predict_text")

        if predict_button and user_text.strip():
            with st.spinner("Analyzing your text..."):
                full_category, category_name = predict_category(user_text, pipeline)
            if full_category and category_name:
                st.success(f"This sentence belongs to **{category_name}**")
        elif predict_button and not user_text.strip():
            st.warning("⚠️ Please enter some text to classify.")

    with tabs[1]:
        st.markdown("### Enter a URL from Reddit, a news article, or a Hugging Face page:")
        url = st.text_input(
            "URL",
            placeholder="https://www.reddit.com/r/... | https://news.site/article | https://huggingface.co/...",
            help="We read only the page meta tags (og:, twitter:, description, title) and classify.",
            key="input_url"
        )
        go = st.button("🌐 Fetch & Predict", type="primary", use_container_width=True, key="predict_url")
        if go:
            if not url.strip():
                st.warning("⚠️ Please paste a URL.")
            else:
                with st.spinner("Fetching page meta and classifying..."):
                    meta_text, meta_fields = fetch_meta_summary(url.strip())
                if not meta_text or len(meta_text.split()) < 5:
                    st.error("Could not read enough meta text from the page. Try a different URL.")
                else:
                    st.caption(f"Meta text length: {len(meta_text.split())} words")
                    with st.spinner("Classifying meta text..."):
                        label, category_name, conf = predict_with_confidence(meta_text, pipeline)
                    if label and category_name:
                        st.success(f"Predicted: **{category_name}**" + (f" (confidence ~ {conf:.2f})" if conf is not None else ""))
                        with st.expander("View extracted meta data", expanded=False):
                            st.write({k: v for k, v in meta_fields.items() if v})
                        # If confidence is low or meta very short, attempt full-content fallback silently
                        needs_fallback = (conf is not None and conf < 0.55) or len(meta_text.split()) < 20
                        if needs_fallback:
                            with st.spinner("Low-confidence meta. Extracting main content for a better read..."):
                                full_text = fetch_text_from_url(url.strip(), meta_only=False)
                            if full_text and len(full_text.split()) >= 40:
                                _, better_name, better_conf = predict_with_confidence(full_text, pipeline)
                                if better_name and (better_conf or 0) > (conf or 0):
                                    st.info(f"Refined prediction from full content: **{better_name}**" + (f" (confidence ~ {better_conf:.2f})" if better_conf is not None else ""))
    
    # Removed duplicate text input and prediction section to avoid duplicate element IDs
    
    # Sidebar with information
    st.sidebar.markdown("### Categories")
    for raw_label, display_name in CATEGORY_DISPLAY_NAMES.items():
        st.sidebar.write(f"{raw_label} → {display_name}")
   
if __name__ == "__main__":
    main()
