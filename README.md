# Boxing Head-to-Head Simulator

**Author:** Jonathan Kennedy  
**Goal:** Predict match outcomes and quantify competitive advantages using real-world fighter data.

## 🥊 Overview
This project applies data analytics and machine learning to a practical decision-support scenario: predicting outcomes between two competitors based on historical performance metrics.  
The model evaluates key factors—height, reach, stance, and win ratio—to estimate win probabilities and identify matchup advantages.  
While built around boxing, the same framework can support business cases like **talent evaluation, risk modeling, or predictive matchups in finance and real estate markets**.

## 📈 Business Context
Sports organizations, analysts, and betting firms use predictive modeling to optimize matchmaking, manage risk, and enhance fan engagement.  
Similarly, this approach mirrors how businesses assess competitors or investments—by comparing measurable attributes to forecast outcomes.  
This project demonstrates how a **data-driven decision system** can transform raw information into actionable predictions.

## 📂 Files
- `train_h2h.py` — Trains the machine learning model with NaN-safe preprocessing and calibrated probabilities.  
- `app_h2h.py` — Streamlit UI for selecting fighters, visualizing stats, and running simulations.  
- `boxing_fighters_master.xlsx` — Fighter attributes and performance metrics.  
- `fights_seed.xlsx` — Historical fight outcomes used for supervised learning.  
- `model_artifacts/` — Trained model and metadata files.

## 🚀 Run Instructions (Windows PowerShell)
```bash
python train_h2h.py --boxers_csv boxing_fighters_master.xlsx --fights_csv fights_seed.xlsx --out_dir model_artifacts
streamlit run app_h2h.py
