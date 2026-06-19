---
title: Market Trend Predictor
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# Financial News Based Market Impact Prediction using FinBERT

Predict whether a stock-related financial news headline is **Bullish**, **Bearish**, or **Neutral** using only the headline text. 

To overcome the low signal-to-noise ratio and ~56% accuracy ceiling of raw next-day stock returns, this project implements a **hybrid dataset architecture** merging clean human-labeled data with historical returns-based labels.

## Datasets Used

This project utilizes a hybrid dataset combination:
1. **Twitter Financial News Sentiment Dataset** ([Hugging Face zero-shot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)):
   - **Gold Labels**: 11,931 human-labeled financial headlines.
   - **Labels**: Bullish, Bearish, and Neutral.
2. **Daily Financial News Analyst Ratings Dataset** ([Kaggle Analyst Ratings processed dataset](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests)):
   - **Silver Labels**: ~1.4M raw financial analyst headlines processed for 6,000+ stocks.
   - **Labels**: Returns-based silver labels mapped via historical stock returns using `yfinance`.


## Project Workflow

1. **Dataset Ingestion**: Load the Kaggle `analyst_ratings_processed.csv` dataset and human-labeled Twitter Financial News dataset.
2. **Exploratory Data Analysis (EDA)**: Run EDA and save charts for missing values, ticker distributions, and monthly publication dates.
3. **Hybrid Label Generation**:
   - Load **Gold Labels** from `zeroshot/twitter-financial-news-sentiment` (~11k clean human-labeled records).
   - Generate **Silver Labels** via `yfinance` next-trading-day returns on the Kaggle CSV news headlines, using adjusted ±2% thresholds.
   - Rebalance the returns-based labels via undersampling of the Neutral class to handle class imbalance.
   - Combine the datasets to construct a robust training set.
4. **Data Splitting**: Partition the clean human-labeled Twitter validation dataset 50/50 into validation and test sets (avoiding evaluation on noisy yfinance returns).
5. **Model Baselines**: Train baseline classifiers (Logistic Regression, Random Forest, and XGBoost) using TF-IDF.
6. **FinBERT Fine-Tuning**: Fine-tune `ProsusAI/finbert` (or `FacebookAI/roberta-base`) using a custom `WeightedTrainer` to handle label imbalance dynamically, incorporating a linear learning rate scheduler with a warmup ratio of 0.06.
7. **Serving & Deployment**: Serve predictions via a Streamlit app deployable locally or to Hugging Face Spaces.

## Repository Structure

```text
.
├── app.py                                    # Streamlit app (local run and HF Spaces)
├── market_impact_training_betterres.ipynb    # Main Jupyter training notebook
├── requirements.txt                          # Comprehensive package installation list
└── README.md                                 # Project documentation
```

## Setup & Local Training

1. Clone this repository:
   ```bash
   git clone https://github.com/Vinilnaik3705/Market-Trend-Predictor.git
   cd Market-Trend-Predictor
   ```

2. Initialize virtual environment and install requirements:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the training notebook `market_impact_training_betterres.ipynb` to download datasets, process labels, train baselines, and fine-tune FinBERT.



---
Developed by **Vinil Naik**
