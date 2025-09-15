
import os
import uuid
import json
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timezone
from PyPDF2 import PdfReader
import praw
from urllib.parse import urlparse


# Load environment variables
load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Hugging Face API token


# Initialize Reddit client if credentials are present
reddit = None
if all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )
    except Exception:
        reddit = None


# ---------------------- Data Fetchers (Link mode) ----------------------
def fetch_reddit_post(url: str) -> dict:
    if reddit is None:
        raise RuntimeError("Reddit API credentials are missing or invalid.")
    submission = reddit.submission(url=url)
    record = {
        "id": str(uuid.uuid4()),
        "source": "reddit",
        "author": submission.author.name if submission.author else "unknown",
        "timestamp": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat(),
        "text": (submission.title or "") + "\n" + (submission.selftext or ""),
        "metadata": {
            "language": "en",
            "likes": submission.score,
            "rating": None,
            "url": url,
        },
    }
    return record


def fetch_news(query: str) -> dict | None:
    if not NEWS_API_KEY:
        raise RuntimeError("NEWS_API_KEY is missing in .env")
    url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(query)}&apiKey={NEWS_API_KEY}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if "articles" not in payload or len(payload["articles"]) == 0:
        return None

    article = payload["articles"][0]
    record = {
        "id": str(uuid.uuid4()),
        "source": "news",
        "author": article.get("author") or "unknown",
        "timestamp": article.get("publishedAt"),
        "text": (article.get("title") or "") + "\n" + (article.get("description") or ""),
        "metadata": {
            "language": article.get("language", "en"),
            "likes": None,
            "rating": None,
            "url": article.get("url"),
        },
    }
    return record


# Helper: normalize HF input to a search-friendly query
def normalize_hf_query(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return text
    # If it's a URL to huggingface.co, extract path components
    try:
        parsed = urlparse(text)
        if parsed.scheme in {"http", "https"} and parsed.netloc.endswith("huggingface.co"):
            path = parsed.path.strip("/")
            if not path:
                return ""
            # Prefer last segment as the model id; if two segments exist (org/model), use that
            segments = [seg for seg in path.split("/") if seg]
            if len(segments) >= 2:
                return f"{segments[-2]}/{segments[-1]}"
            return segments[-1]
    except Exception:
        pass
    return text


# Existing single-result helper retained for compatibility
def fetch_huggingface_models(query: str) -> dict | None:
    q = normalize_hf_query(query)
    url = f"https://huggingface.co/api/models?search={requests.utils.quote(q)}"
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code == 401 and headers:
        # Retry without token if unauthorized
        response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, list) or len(payload) == 0:
        return None

    model = payload[0]
    model_id = model.get("modelId") or model.get("id") or "unknown"
    record = {
        "id": str(uuid.uuid4()),
        "source": "huggingface",
        "author": model.get("author") or "unknown",
        "timestamp": model.get("lastModified") or None,
        "text": (model_id or "") + "\n" + (model.get("pipeline_tag") or ""),
        "metadata": {
            "language": "en",
            "likes": None,
            "rating": None,
            "url": f"https://huggingface.co/{model_id}",
        },
    }
    return record


# New: fetch top-N models
def fetch_hf_top_n(query: str, limit: int) -> list[dict]:
    q = normalize_hf_query(query)
    url = f"https://huggingface.co/api/models?search={requests.utils.quote(q)}"
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code == 401 and headers:
        # Retry without token if unauthorized
        response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()

    results: list[dict] = []
    if isinstance(payload, list):
        for model in payload[: max(1, int(limit))]:
            model_id = model.get("modelId") or model.get("id") or "unknown"
            results.append(
                {
                    "id": str(uuid.uuid4()),
                    "source": "huggingface",
                    "author": model.get("author") or "unknown",
                    "timestamp": model.get("lastModified") or None,
                    "text": (model_id or "") + "\n" + (model.get("pipeline_tag") or ""),
                    "metadata": {
                        "language": "en",
                        "likes": None,
                        "rating": None,
                        "url": f"https://huggingface.co/{model_id}",
                    },
                }
            )
    return results


# ---------------------- Persistence Helpers ----------------------
OUTPUT_JSON = "output.json"  # append link data here
PDF_CSV = "data_store.csv"   # store pdf extraction here


def append_record_to_output_json(record: dict) -> None:
    existing: list[dict] = []
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                existing = json.load(f) or []
                if not isinstance(existing, list):
                    existing = []
        except Exception:
            existing = []
    existing.append(record)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)


def append_many_to_output_json(records: list[dict]) -> None:
    existing: list[dict] = []
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                existing = json.load(f) or []
                if not isinstance(existing, list):
                    existing = []
        except Exception:
            existing = []
    existing.extend(records)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)


# Write pretty JSON object per record to data_store.csv for PDFs
def append_pdf_record_pretty_json(filename: str, content: str) -> None:
    record = {
        "id": int(datetime.now().timestamp()),
        "filename": filename,
        "source_type": "file",
        "file_type": "pdf",
        "content": content or "",
        "upload_time": datetime.now().isoformat(),
    }
    pretty = json.dumps(record, ensure_ascii=False, indent=4)
    with open(PDF_CSV, "a", encoding="utf-8") as f:
        f.write(pretty)
        f.write("\n\n")


def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    pages_text: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)
    return "\n".join(pages_text).strip()


# ---------------------- Streamlit UI ----------------------
st.title("📊 NarrativeNexus - Link/PDF Collector")
st.write("Choose to fetch from a link source (Reddit, News, HuggingFace) or upload a PDF for text extraction.")

mode = st.selectbox("Select input type:", ["Link", "PDF"])

if mode == "Link":
    st.subheader("Link Source")
    link_source = st.selectbox("Choose Source:", ["Reddit Post", "News Article", "Hugging Face Model"]) 
    user_input = st.text_input("Enter Reddit URL / News query / HuggingFace search term (or model URL):")

    # Allow selecting number of results for HF
    hf_limit = 5
    if link_source == "Hugging Face Model":
        hf_limit = st.number_input("Number of models to fetch", min_value=1, max_value=50, value=5, step=1)

    if st.button("Fetch & Append to output.json"):
        if not user_input.strip():
            st.error("Please provide a valid input.")
        else:
            try:
                record = None
                if link_source == "Reddit Post":
                    record = fetch_reddit_post(user_input)
                    if record:
                        append_record_to_output_json(record)
                        st.success("Appended 1 item to output.json")
                        st.subheader("Preview")
                        st.json(record)
                elif link_source == "News Article":
                    record = fetch_news(user_input)
                    if record is None:
                        st.error("No news found for this query.")
                    else:
                        append_record_to_output_json(record)
                        st.success("Appended 1 item to output.json")
                        st.subheader("Preview")
                        st.json(record)
                elif link_source == "Hugging Face Model":
                    records = fetch_hf_top_n(user_input, int(hf_limit))
                    if not records:
                        st.error("No Hugging Face models found for this query.")
                    else:
                        append_many_to_output_json(records)
                        st.success(f"Appended {len(records)} item(s) to output.json")
                        st.subheader("Preview (first)")
                        st.json(records[0])
            except Exception as e:
                st.error(f"Error: {e}")

else:  # PDF mode
    st.subheader("PDF Upload")
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    if st.button("Extract & Save to data_store.csv"):
        if uploaded_pdf is None:
            st.error("Please upload a PDF file.")
        else:
            try:
                text_content = extract_text_from_pdf(uploaded_pdf)
                if not text_content:
                    st.warning("No text could be extracted from the PDF.")
                # Append as pretty JSON (indented) into data_store.csv
                append_pdf_record_pretty_json(uploaded_pdf.name, text_content)
                st.success("Saved to data_store.csv")
                st.subheader("Extracted Text (Preview)")
                st.write(text_content[:2000] + ("..." if len(text_content) > 2000 else ""))
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
