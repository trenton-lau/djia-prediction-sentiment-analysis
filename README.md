
# STAT3011_2324_GP9: Stock Market Price Prediction

**(!!! EDIT THIS README THOROUGHLY !!!)**

This project explores stock market price prediction using various techniques including LSTM, NLP sentiment analysis, and data preprocessing.

## Project Goal

*(Describe the main objective of the project in detail.)*

## Directory Structure


├── .gitignore # Files ignored by Git
├── README.md # This file - project overview (NEEDS UPDATE)
├── requirements.txt # Python dependencies (Generate using 'pip freeze > requirements.txt')
├── renv.lock # R dependencies (Generate using 'renv::snapshot()')
│
├── data/ # Data files (or instructions)
│ ├── README_DATA.md # Create this: Explain data sources
│ └── ... (data files like CSVs - if small enough)
│
├── notebooks/ # Jupyter Notebooks for analysis and modeling
│ ├── 0_Case_Replication.ipynb
│ ├── 1_Window_Expansion.ipynb # Note: Needs manual creation from scripts/py/window_expansion.py
│ ├── 2_NLP_Analysis_Setup.ipynb # Create this? Narrative for R/SQL steps
│ ├── 3_LSTM_Sentiment_Analysis.ipynb
│ ├── 4_LSTM_Transformer_Exploration.ipynb
│ └── utils/ # Developmental/scratch notebooks
│ └── LSTM_Initial_Dev_part1.ipynb
│ └── LSTM_Initial_Dev_part2.ipynb
│ └── LSTM_Hyperparameter_Tuning.ipynb # Note: Needs manual creation from scripts/py/lstm_hyperparameter_tuning.py
│
├── scripts/ # Reusable or standalone scripts
│ ├── py/ # Python Scripts
│ │ ├── datapreparation.py
│ │ ├── getdata_NYtimes.py
│ │ ├── model_lstm_basic.py
│ │ ├── model_lstm_final_draft.py
│ │ ├── lstm_hyperparameter_tuning.py # Source for manual notebook creation
│ │ └── window_expansion.py # Source for manual notebook creation
│ ├── R/ # R Scripts (NLP steps)
│ │ ├── 1_process_data.R
│ │ ├── 2_generate_wordclouds.R
│ │ ├── 3_evaluate_sentiment_R.R
│ │ └── 4_evaluate_sentiment_Py.R
│ └── sql/ # SQL Scripts
│ └── calculate_lag_change.sql
│
└── reports/ # Supporting documents
├── Group9_Presentation.pptx
└── Reference_Paper.pdf

## Data

*(**Crucial:** Explain where ALL data comes from. Provide links, generation script steps, or source info. DO NOT commit large files like `.pkl`)*
*   Create `data/README_DATA.md` with details.

## Setup

*(Provide detailed, step-by-step setup instructions)*

1.  **Clone:** `git clone <your-repo-url>`
2.  **Navigate:** `cd <repo-name>`
3.  **Python Dependencies:**
    *   *(Ensure you have Python 3.x installed)*
    *   *(Consider creating a virtual environment: `python -m venv env` then `source env/bin/activate` or `.\env\Scriptsctivate`)*
    *   Run `pip install -r requirements.txt` (**You need to create this file!**)
4.  **R Dependencies:**
    *   *(Ensure you have R installed)*
    *   *(Install renv: `install.packages("renv")` in R)*
    *   Run `renv::restore()` in R console within the project directory (**You need to run `renv::init()` and `renv::snapshot()` first!**)
5.  **Data:** Follow instructions in `data/README_DATA.md`.
6.  **NLTK Data (if applicable):** May need `nltk.download('vader_lexicon')` etc. in Python.

## Usage

*(Explain how to run the key parts of the project)*
*   Example: Run notebooks in numerical order in the `notebooks/` directory.
*   Example: Execute R scripts in `scripts/R/` sequentially.

## Results

*(Summarize key findings or point to notebooks/reports with results)*

## References

*   *(List any papers, articles, or significant libraries used)*
