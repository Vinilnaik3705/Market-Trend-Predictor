# Financial News Based Market Impact Prediction using FinBERT

Predict whether a stock-related financial news headline is **Bullish**, **Bearish**, or **Neutral** using only the headline text. Labels are generated from next-trading-day stock returns, and models include TF-IDF baselines plus a fine-tuned FinBERT classifier.

## Project Workflow

1. Load and understand the Kaggle `analyst_ratings_processed.csv` dataset.
2. Run EDA and save charts for missing values, duplicate headlines, ticker concentration, and publication dates.
3. Generate market-impact labels by joining each headline to historical prices from `yfinance`.
4. Clean headline text and create train/validation/test splits.
5. Train TF-IDF baselines: Logistic Regression, Random Forest, and XGBoost.
6. Fine-tune `ProsusAI/finbert` for 3-class classification.
7. Analyze model errors and compare baseline vs FinBERT performance.
8. Serve inference through a Streamlit app.

## Repository Structure

```text
.
├── app/
│   └── streamlit_app.py
├── config/
│   └── config.yaml
├── docs/
│   ├── architecture_diagram.md
│   ├── project_report.md
│   ├── resume_description.md
│   └── workflow_diagram.md
├── scripts/
│   ├── 01_eda.py
│   ├── 02_generate_labels.py
│   ├── 03_prepare_splits.py
│   ├── 04_train_baselines.py
│   ├── 05_train_finbert.py
│   ├── 06_error_analysis.py
│   └── 07_predict.py
├── src/
│   └── market_impact/
│       ├── baseline.py
│       ├── config.py
│       ├── data.py
│       ├── finbert.py
│       ├── inference.py
│       ├── labeling.py
│       ├── metrics.py
│       ├── preprocessing.py
│       └── visualization.py
├── tests/
│   ├── test_labeling.py
│   └── test_preprocessing.py
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place the Kaggle ZIP or extracted CSV path in `config/config.yaml`:

```yaml
raw_zip_path: "C:/Users/VINIL NAIK/OneDrive/Desktop/stock predictor using news/analyst_ratings_processed.csv.zip"
```

## Run the Pipeline

```bash
python scripts/01_eda.py --config config/config.yaml
python scripts/02_generate_labels.py --config config/config.yaml
python scripts/03_prepare_splits.py --config config/config.yaml
python scripts/04_train_baselines.py --config config/config.yaml
python scripts/05_train_finbert.py --config config/config.yaml
python scripts/06_error_analysis.py --config config/config.yaml
```

The generated labeled dataset is saved as:

```text
data/processed/market_impact_dataset.csv
```

## Inference

```bash
python scripts/07_predict.py --headline "NVIDIA signs $50 billion AI infrastructure deal"
```

Example output:

```json
{
  "headline": "NVIDIA signs $50 billion AI infrastructure deal",
  "predicted_class": "Bullish",
  "confidence": 0.84,
  "probabilities": {
    "Bullish": 0.84,
    "Bearish": 0.05,
    "Neutral": 0.11
  }
}
```

## Streamlit Deployment

```bash
streamlit run app/streamlit_app.py
```

The app lets users enter a financial headline and returns Bullish/Bearish/Neutral probabilities, predicted class, confidence score, and model metadata.

## Notes

- Labels are generated from historical prices, so `yfinance` requires internet access.
- FinBERT fine-tuning is GPU-friendly but can run on CPU with smaller `max_train_samples`.
- XGBoost is included as a baseline dependency; if unavailable, the baseline script records the skip reason instead of failing the full run.
