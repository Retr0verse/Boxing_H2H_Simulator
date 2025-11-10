# Boxing H2H — Fixed, One-and-Done

## Files
- `train_h2h.py` — NaN-safe training, calibrated probabilities, saves `model_artifacts/pairwise_model.joblib`.
- `app_h2h.py` — Streamlit UI with modulo bug fixed and NaN-safe feature engineering.

## Run (Windows PowerShell)
```
python train_h2h.py --boxers_csv boxing_fighters_master.csv --fights_csv fights_seed.csv --out_dir model_artifacts
streamlit run app_h2h.py
```
Sidebar paths:
- Fighters CSV: `boxing_fighters_master.csv`
- Fights CSV: `fights_seed.csv`
- Model file: `model_artifacts/pairwise_model.joblib`
