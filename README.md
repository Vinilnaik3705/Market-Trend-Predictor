# Financial News Based Market Impact Prediction using FinBERT

Predict whether a stock-related financial news headline is **Bullish**, **Bearish**, or **Neutral** using only the headline text. 

To overcome the low signal-to-noise ratio and ~56% accuracy ceiling of raw next-day stock returns, this project implements a **hybrid dataset architecture** merging clean human-labeled data with historical returns-based labels.

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
├── app/
│   └── streamlit_app.py                      # Local Streamlit server script
├── app.py                                    # Entrypoint app.py for Hugging Face Spaces
├── hf_requirements.txt                       # Optimized requirements file for HF Spaces
├── market_impact_training_betterres.ipynb    # Main Jupyter training notebook
├── requirements.txt                          # Comprehensive package installation list
├── pyproject.toml                            # Package configuration and specifications
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

## Streamlit App Deployment (Local)

To launch the local web server:
```bash
streamlit run app/streamlit_app.py
```

## Hugging Face Spaces Deployment

For direct deployment to Hugging Face Spaces:
1. Create a new Streamlit Space at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Upload the following files to your space repository:
   - `app.py`
   - `hf_requirements.txt` (rename this to `requirements.txt` when pushing)
3. Upload your fine-tuned model to Hugging Face Hub:
   ```python
   model.push_to_hub("username/finbert-market-impact")
   tokenizer.push_to_hub("username/finbert-market-impact")
   ```
4. Enter your model repository ID (`username/finbert-market-impact`) in the Space sidebar to load and serve predictions.
