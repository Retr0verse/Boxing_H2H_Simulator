import argparse, os, json, joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

RANDOM_STATE = 123
CORE_NUMERIC = ['age','height_cm','reach_cm','wins','losses','draws','ko_wins',
                'fights_last_24mo','days_since_last_fight','ko_rate','reach_to_height']

def engineer_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CORE_NUMERIC:
        if c not in out.columns:
            out[c] = 0.0
    out[CORE_NUMERIC] = out[CORE_NUMERIC].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    out['is_southpaw'] = out.get('stance','').astype(str).str.lower().str.contains('southpaw').astype(int)
    out['is_switch']   = out.get('stance','').astype(str).str.lower().str.contains('switch').astype(int)
    return out

def make_pairwise_dataset(boxers: pd.DataFrame, fights: pd.DataFrame):
    bx = engineer_numeric(boxers).set_index('fighter')
    num_cols = CORE_NUMERIC + ['is_southpaw','is_switch']
    rows, y = [], []
    for _, r in fights.iterrows():
        a, b = r['fighter_a'], r['fighter_b']
        if a not in bx.index or b not in bx.index:
            continue
        va, vb = bx.loc[a, num_cols].values.astype(float), bx.loc[b, num_cols].values.astype(float)
        rows.append(va - vb); y.append(1)
        rows.append(vb - va); y.append(0)
    X = pd.DataFrame(rows, columns=[f'diff_{c}' for c in num_cols]).fillna(0.0)
    y = pd.Series(y, name='y')
    return X, y, num_cols

def evaluate(clf, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')[:,1]
    pred  = (proba>=0.5).astype(int)
    auc   = roc_auc_score(y, proba)
    acc   = accuracy_score(y, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    brier = brier_score_loss(y, proba)
    return {'AUC': float(auc), 'Accuracy': float(acc), 'Precision': float(prec), 'Recall': float(rec), 'F1': float(f1), 'Brier': float(brier)}

def main(boxers_csv, fights_csv, out_dir):
    boxers = pd.read_csv(boxers_csv)
    fights = pd.read_csv(fights_csv)
    X, y, num_cols = make_pairwise_dataset(boxers, fights)

    models = {
        'GB': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'RF': RandomForestClassifier(n_estimators=800, random_state=RANDOM_STATE, n_jobs=-1),
        'LR': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    }
    results = {}
    best_name, best_auc, best_model = None, -1.0, None
    for name, m in models.items():
        metrics = evaluate(m, X, y)
        results[name] = metrics
        if metrics['AUC'] > best_auc:
            best_name, best_auc, best_model = name, metrics['AUC'], m

    best_model.fit(X, y)
    cal = CalibratedClassifierCV(best_model, cv=5, method='isotonic')
    cal.fit(X, y)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({'model': cal, 'features': [f'diff_{c}' for c in num_cols], 'src_numeric': num_cols},
                os.path.join(out_dir, 'pairwise_model.joblib'))
    with open(os.path.join(out_dir, 'cv_report.json'), 'w') as f:
        json.dump({'candidates': results, 'selected': best_name}, f, indent=2)
    print('Saved to', os.path.join(out_dir, 'pairwise_model.joblib'))
    print('Selected:', best_name, 'AUC', best_auc)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--boxers_csv', required=True)
    ap.add_argument('--fights_csv', required=True)
    ap.add_argument('--out_dir', default='model_artifacts')
    args = ap.parse_args()
    main(args.boxers_csv, args.fights_csv, args.out_dir)
