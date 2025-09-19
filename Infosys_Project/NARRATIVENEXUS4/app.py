import streamlit as st
import joblib
import os
import sys
from pathlib import Path
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
    """Extract meaningful category names with special handling for specific categories"""
    # Special cases for better category names
    category_mapping = {
        'comp.os.ms-windows.misc': 'os windows',
        'comp.sys.ibm.pc.hardware': 'pc hardware',
        'comp.sys.mac.hardware': 'mac hardware',
        'comp.windows.x': 'windows',
        'talk.politics.mideast': 'mideast politics',
        'talk.politics.misc': 'politics',
        'talk.religion.misc': 'religion'
    }
    
    # Check if it's a special case
    if full_category in category_mapping:
        return category_mapping[full_category]
    
    # For others, return the last word
    return full_category.split('.')[-1]

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
    st.markdown("### Enter a sentence to classify:")
    
    # Text input
    user_text = st.text_area(
        "Type your sentence here:",
        placeholder="Example: The new graphics card from NVIDIA has amazing performance for 3D rendering.",
        height=100,
        help="Enter any text and click 'Predict Category' to see which category it belongs to."
    )
    
    # Predict button
    predict_button = st.button("🔍 Predict Category", type="primary", use_container_width=True)
    
    # Handle prediction when button is clicked
    if predict_button and user_text.strip():
        with st.spinner("Analyzing your text..."):
            full_category, category_name = predict_category(user_text, pipeline)
        
        if full_category and category_name:
            # Simple result display without boxes
            st.success(f"This sentence belongs to **{category_name}**")
    
    elif predict_button and not user_text.strip():
        st.warning("⚠️ Please enter some text to classify.")
    
    # Sidebar with information
   
if __name__ == "__main__":
    main()
