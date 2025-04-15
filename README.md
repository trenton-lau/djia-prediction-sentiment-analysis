# Stock Market Price Prediction

## Project Overview

This project tackles the prediction of the Dow Jones Industrial Average (DJIA) adjusted closing price using historical price data and sentiment analysis derived from news headlines. The analysis explores several machine learning approaches, including traditional methods, Natural Language Processing (NLP) techniques, and deep learning with Long Short-Term Memory (LSTM) networks. The project workflow progresses through case replication, methodology improvements via window expansion and NLP, and finally, culminates in an LSTM-based model incorporating sentiment features.

---

## Project Workflow & Methodology

The project is divided into four main stages:

1.  **01 Replication:** Replicating baseline stock prediction models using Random Forest, Logistic Regression, and Multilayer Perceptron (MLP). *Adjustments were made to handle outdated code from original sources.*
2.  **02 Window Expansion:** Improving baseline models by extending the training dataset's historical time window (from 8 years/10 months per year to 9 years/11 months per year) to include more past observations, aiming for increased robustness and potentially higher complexity handling. Performance is measured by Mean Squared Error (MSE) improvement.
3.  **03 NLP Sentiment Analysis:** Incorporating sentiment analysis of 10 years of news headlines (likely from the New York Times API) to gauge market sentiment. This involved comparing R and Python (NLTK) sentiment packages and analyzing their performance in predicting stock price direction (increase/decrease) using confusion matrices. Explores the limitations of lexicon-based sentiment analysis.
4.  **04 LSTM Implementation:** Utilizing Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network (RNN) well-suited for sequence data like time series. LSTMs are designed to remember information over long periods, crucial for capturing market context. This phase introduces sentiment scores and a BIAS indicator (percentage difference between price and its moving average) as additional factors in the LSTM model, analyzing the impact of different BIAS window sizes.

## Key Concepts & Models

*   **Prediction Target:** Adjusted Close Price (Adj Close) of the DJIA index (preferred over Close Price as it accounts for dividends/splits, acting as a better measure and potential upper bound).
*   **Baseline Models (Replication):** Random Forest Regressor, Logistic Regression (adapted for rise/fall classification), Multilayer Perceptron (MLP).
*   **Improvement Techniques:**
    *   **Window Expansion:** Increasing training data history (results showed MSE reduction).
    *   **Alignment & Smoothing:** Techniques (offsetting, EWMA) applied, particularly to Random Forest predictions, to better match magnitude and reduce noise.
*   **NLP:** Lexicon-based sentiment analysis (NLTK `sentwordnet`, R `sentiment` package) extracting Positive, Negative, Neutral, and Compound scores. Focus on limitations (e.g., handling double negatives, relevance of news).
*   **LSTM:** Deep learning model for sequence prediction, leveraging its "memory" to understand context. Factors included Sentiment Score and BIAS indicator.

## Directory Structure


.
├── .gitignore 
├── README.md # This project overview file (NEEDS FINAL REVIEW)
├── requirements.txt # Python dependencies (Generate via 'pip freeze')
├── renv.lock # (Optional) R dependencies via renv snapshot
│
├── data/ # Data files (or instructions)
│ ├── README_DATA.md # Create this: Explain DJIA / NYTimes data source/acquisition
│ ├── DJIA_data.csv # Example source data file
│ └── # Add other relevant data files or .pkl paths (use .gitignore for large files)
│
├── notebooks/ # Jupyter Notebooks for analysis and modeling
│ ├── 0_Case_Replication.ipynb # Contains RF, LogReg, MLP baseline implementation (if applicable)
│ ├── 1_Window_Expansion.ipynb # Implementation showing window expansion effect (likely scripts/py/window_expansion.py content)
│ ├── 2_NLP_Analysis_Setup.ipynb # (Optional) Setup/details for R/Python NLP comparison
│ ├── 3_LSTM_Sentiment_Analysis.ipynb # Main LSTM model incorporating sentiment/BIAS
│ ├── 4_LSTM_Transformer_Exploration.ipynb # (Optional) Advanced exploration
│ └── utils/ # Developmental/scratch notebooks
│ ├── LSTM_Initial_Dev_part1.ipynb # Initial LSTM setup
│ └── LSTM_Initial_Dev_part2.ipynb # Further LSTM development
│ └── # LSTM_Hyperparameter_Tuning.ipynb (content from scripts/py/lstm_hyperparameter_tuning.py)
│
├── scripts/ # Reusable or standalone scripts
│ ├── py/ # Python Scripts
│ │ ├── datapreparation.py # Script for preparing DJIA and NYTimes data
│ │ ├── getdata_NYtimes.py # Script for fetching NYTimes data via API
│ │ ├── model_lstm_basic.py # Basic LSTM implementation
│ │ ├── model_lstm_final_draft.py # Refined LSTM with evaluation metrics
│ │ ├── lstm_hyperparameter_tuning.py # Script for hyperparameter search (may be integrated into notebook)
│ │ └── window_expansion.py # Script showing window expansion effects (may be integrated into notebook)
│ ├── R/ # R Scripts (NLP steps)
│ │ ├── # (Assumed based on presentation) - Need script naming conventions
│ │ ├── 1_process_data.R # Initial R data prep for NLP
│ │ ├── 2_generate_wordclouds.R # R word cloud generation
│ │ ├── 3_evaluate_sentiment_R.R # R sentiment scoring and confusion matrix
│ │ └── 4_evaluate_sentiment_Py.R # Confusion matrix using Python scores via R
│ └── sql/ # (If SQL was used for intermediate steps)
│ └── calculate_lag_change.sql # Example from previous project
│
└── reports/ # Supporting documents
└── Reference_Paper.pdf # Referenced paper on LSTM+news impact

## Data Sources

*   **DJIA Data:** Historical DJIA index price data (Open, High, Low, Close, Adj Close, Volume). *(Specify source if known, e.g., Yahoo Finance, or if included as `DJIA_data.csv`)*. Data covers the period required for training (e.g., 2007-2015) and testing (e.g., 2016).
*   **News Data:** 10 years of news headlines, likely sourced from the New York Times Archive API using `scripts/py/getdata_NYtimes.py`. *(Specify date range, e.g., 2007-2016)*. The `scripts/py/datapreparation.py` script processes and merges this with the DJIA data. The resulting processed file (`pickled_ten_year_filtered_lead_para.pkl`) should ideally be generated locally and gitignored if large.

## Setup

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/trenton-lau/djia-prediction-sentiment-analysis.git
    cd djia-prediction-sentiment-analysis
    ```
2.  **Create Environment (Recommended):** Use `venv` or `conda`.
3.  **Install Python Dependencies:** Generate and use `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *(Likely includes: `pandas`, `numpy`, `scikit-learn`, `tensorflow`/`keras`, `nltk`, `matplotlib`, `pywin32` (for PPTX conversion), `requests`, `pyperclip`, `treeinterpreter`)*
4.  **Install R Dependencies (If running R scripts):** Ensure R is installed. Use `renv` or manually install required packages (`tm`, `sentimentr`, `dplyr`, `ggplot2`, `tidytext`, `wordcloud`, `RColorBrewer`, `readxl`, `stringr`, `forcats`, `caret`, etc.).
5.  **Download NLTK Data:** Run required downloads within Python/notebooks:
    ```python
    import nltk
    nltk.download('vader_lexicon')
    nltk.download('wordnet') # Likely needed by sentiment tools
    nltk.download('punkt')   # Likely needed for tokenization
    # Add others as needed
    ```
6.  **Acquire Data:** Follow instructions in `data/README_DATA.md` (or run `getdata_NYtimes.py` if API key is configured) to obtain DJIA and NYTimes data.
7.  **Prepare Data:** Run `scripts/py/datapreparation.py` to generate the merged `.pkl` file used by LSTM notebooks.

## Usage

*   **Data Prep:** Ensure data acquisition and preparation scripts/notebooks are run first.
*   **Replication & Improvements:** Run relevant notebooks/scripts (e.g., from `notebooks/0*`, `scripts/py/window_expansion.py`) to see baseline and window expansion results.
*   **NLP Analysis:** Execute R scripts in `scripts/R/` (1-4) sequentially for sentiment comparison and word cloud generation.
*   **LSTM Modeling:** Run notebooks `notebooks/utils/LSTM*` and `notebooks/3*`, `notebooks/4*` to train and evaluate the LSTM models with sentiment/BIAS factors.

## Results Summary

*   **Replication:** Baseline models (RF, LogReg, MLP) established initial performance. MLP performed best among the three baseline replications.
*   **Window Expansion:** Extending the training data from 8 years (10 months/year) to 9 years (11 months/year) **reduced the Mean Squared Error (MSE)** across all tested models (RF, LogReg, MLP), indicating that more historical data improved predictive accuracy. *
*   **NLP Sentiment Analysis:**
    *   Lexicon-based methods (R `sentiment`/bing, Python NLTK/Vader) were used to score news headlines.
    *   Top positive words included `right`, `trump`, `work`, `win`, `top`, `lead`, `like`, `gain`, `support`, `good`.
    *   Top negative words included `kill`, `attack`, `death`, `protest`, `dead`, `fall`, `loss`, `risk`, `critic`, `debt`.
    *   Confusion matrices showed **accuracy barely above 50%** (R: 51.65%, Python: 51.71%) for predicting stock price direction (Increase/Decrease) based *solely* on the compound sentiment score. R package was slightly better on False Negatives, Python slightly better on False Positives.
    *   **Conclusion:** Basic sentiment analysis alone is a poor predictor, likely due to context issues (double negatives) and irrelevance of many news items (e.g., "Sunday's Breakfast Menu").
*   **LSTM Modeling:**
    *   Introduced as a better approach due to its ability to learn long-term dependencies and weigh feature importance via back-propagation.
    *   Incorporated Sentiment Score and BIAS indicator as factors.
    *   Analysis of different BIAS **window sizes** showed `Window Size = 3` achieved the lowest Training and Testing MSE among sizes 1-5.
    *   The model with `Window Size = 2` provided a representative plot of predicted vs actual values.
    *   Validation loss curves showed potential under-fitting (large gap, slow convergence), well-fitting, and over-fitting (validation loss increases) depending on training duration/parameters.

## Conclusion & Discussion

The project successfully replicated baseline models, demonstrated the benefit of expanding the training window size (reduced MSE), and thoroughly investigated the utility of NLP sentiment analysis. While NLP sentiment derived from news headlines contains relevant signals (top words reflect market-moving themes), **lexicon-based sentiment scores alone proved to be weak predictors of daily stock movement** due to context insensitivity and news irrelevance.

The **LSTM approach represents a significant improvement**. By design, it can better capture temporal dependencies in price data. Incorporating **sentiment scores** and the **BIAS indicator** as engineered features provides the LSTM with richer contextual information beyond just price history. The results showed **acceptable (though not perfect) performance** when using Sentiment Score as a factor (MSEs ~0.0003), and further refinement using the BIAS indicator with optimized window sizes (Window Size = 3 appeared optimal in tests) likely enhances predictions.

**Key Takeaways:**

*   More data (longer time windows) generally improves time-series prediction accuracy.
*   Simple sentiment analysis has limitations for direct stock prediction but can be a useful *feature* within a more complex model.
*   LSTMs are well-suited for financial time series due to their memory capabilities.
*   Feature engineering (like the BIAS indicator) and careful model tuning (like selecting the BIAS window size) are critical for LSTM performance.


## References

*   Ren, Y., Liao, F. and Gong, Y. (2020) 'Impact of news on the trend of stock price change: An analysis based on the deep bidirectiona LSTM model', *Procedia Computer Science*, 174, pp. 128–140. doi:[10.1016/j.procs.2020.06.068](https://doi.org/10.1016/j.procs.2020.06.068).
*   Thanaki, Jalaj. *Machine Learning Solutions*. 1st edition. Packt Publishing, 2018. Print.
