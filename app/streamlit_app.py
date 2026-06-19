from __future__ import annotations

import re
from html import unescape
from pathlib import Path
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Root path of the project
ROOT = Path(__file__).resolve().parents[1]

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
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
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
def load_predictor(model_dir: str):
    return FinBERTMarketImpactPredictor(model_dir)

st.set_page_config(page_title="Market Impact Predictor", page_icon="📈", layout="centered")
st.title("Financial News Market Impact Prediction")

default_model_dir = ROOT / "artifacts" / "finbert" / "best_model"
model_dir = st.sidebar.text_input("Model directory", value=str(default_model_dir))
st.sidebar.markdown("Model: `FacebookAI/roberta-base` fine-tuned for Bullish/Bearish/Neutral classification.")

headline = st.text_area(
    "Financial headline",
    value="NVIDIA signs $50 billion AI infrastructure deal",
    height=100,
)

if st.button("Predict", type="primary"):
    if not headline.strip():
        st.warning("Enter a headline first.")
    elif not Path(model_dir).exists():
        st.error("Fine-tuned model directory not found. Run the training notebook first.")
    else:
        predictor = load_predictor(model_dir)
        result = predictor.predict(headline)
        st.subheader(result["predicted_class"])
        st.metric("Confidence", f"{result['confidence']:.1%}")
        probs = pd.DataFrame(
            [{"Class": k, "Probability": v} for k, v in result["probabilities"].items()]
        ).sort_values("Probability", ascending=False)
        st.bar_chart(probs.set_index("Class"))
        st.dataframe(probs.assign(Probability=lambda d: d["Probability"].map(lambda x: f"{x:.2%}")))
