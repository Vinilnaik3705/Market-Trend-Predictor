from __future__ import annotations

import re
from html import unescape
from pathlib import Path
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Re-implement cleaner locally to make the app self-contained
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
SPECIAL_RE = re.compile(r"[^a-z0-9$%&+\-./\s]")
SPACE_RE = re.compile(r"\s+")

def clean_headline(text: str) -> str:
    """Clean text while keeping finance-relevant symbols like $, %, +, -, and ticker dots."""
    if text is None:
        return ""
    text = HTML_TAG_RE.sub(" ", unescape(str(text)))
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = SPECIAL_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text

class FinBERTMarketImpactPredictor:
    def __init__(self, model_identifier: str):
        self.model_identifier = model_identifier
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_identifier)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_identifier)
        self.model.eval()

    @torch.no_grad()
    def predict(self, headline: str) -> dict:
        inputs = self.tokenizer(headline, return_tensors="pt", truncation=True, max_length=96)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        id2label = self.model.config.id2label
        probabilities = {id2label[i]: float(probs[i]) for i in range(len(probs))}
        predicted_class = max(probabilities, key=probabilities.get)
        return {
            "headline": headline,
            "predicted_class": predicted_class,
            "confidence": probabilities[predicted_class],
            "probabilities": probabilities,
        }

@st.cache_resource
def load_predictor(model_identifier: str):
    return FinBERTMarketImpactPredictor(model_identifier)

st.set_page_config(page_title="Market Impact Predictor", page_icon="📈", layout="centered")
st.title("Financial News Market Impact Prediction")

# Model configuration sidebar
st.sidebar.subheader("Model Configuration")
local_model_path = Path("artifacts/finbert/best_model")
default_model = str(local_model_path) if local_model_path.exists() else "ProsusAI/finbert"

model_identifier = st.sidebar.text_input(
    "Model Path or HF Model ID",
    value=default_model,
    help="Provide a local directory path (e.g., 'artifacts/finbert/best_model') or a Hugging Face Hub Model ID (e.g., 'ProsusAI/finbert')."
)

st.sidebar.markdown("""
### Instructions for custom models:
1. Fine-tune your model using the notebook.
2. Push your model to Hugging Face Hub:
   ```python
   model.push_to_hub("username/finbert-market-impact")
   tokenizer.push_to_hub("username/finbert-market-impact")
   ```
3. Enter your Model ID in the text input above.
""")

headline = st.text_area(
    "Financial headline",
    value="NVIDIA signs $50 billion AI infrastructure deal",
    height=100,
)

if st.button("Predict", type="primary"):
    if not headline.strip():
        st.warning("Enter a headline first.")
    else:
        with st.spinner("Loading model and predicting..."):
            try:
                predictor = load_predictor(model_identifier)
                result = predictor.predict(headline)
                
                st.subheader(result["predicted_class"])
                st.metric("Confidence", f"{result['confidence']:.1%}")
                
                probs = pd.DataFrame(
                    [{"Class": k, "Probability": v} for k, v in result["probabilities"].items()]
                ).sort_values("Probability", ascending=False)
                st.bar_chart(probs.set_index("Class"))
                st.dataframe(probs.assign(Probability=lambda d: d["Probability"].map(lambda x: f"{x:.2%}")))
            except Exception as e:
                st.error(f"Error loading model or running prediction: {e}")
