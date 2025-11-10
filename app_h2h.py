import os, joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Boxing Head-to-Head (Advanced)', layout='wide')
st.title('Boxing Head-to-Head (Advanced)')
st.caption('Load fighters CSV + fights CSV. Train a calibrated pairwise model and simulate matchups.')

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

with st.sidebar:
    st.header('Data')
    fighters_path = st.text_input('Fighters CSV', value='boxing_fighters_master.csv')
    fights_path   = st.text_input('Fights CSV', value='fights_seed.csv')
    artifacts_path= st.text_input('Model file', value='model_artifacts/pairwise_model.joblib')
    train_btn = st.button('Train / Retrain model')

CORE = ['age','height_cm','reach_cm','wins','losses','draws','ko_wins',
        'fights_last_24mo','days_since_last_fight','ko_rate','reach_to_height']

def engineer_numeric(df: pd.DataFrame):
    out = df.copy()
    for c in CORE:
        if c not in out.columns:
            out[c] = 0.0
    out[CORE] = out[CORE].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    out['is_southpaw'] = out.get('stance','').astype(str).str.lower().str.contains('southpaw').astype(int)
    out['is_switch']   = out.get('stance','').astype(str).str.lower().str.contains('switch').astype(int)
    return out, CORE + ['is_southpaw','is_switch']

def make_diff(a_row, b_row, num_cols):
    va = a_row[num_cols].values.astype(float)
    vb = b_row[num_cols].values.astype(float)
    diff = (va - vb).reshape(1,-1)
    return pd.DataFrame(diff, columns=[f'diff_{c}' for c in num_cols])

def train_model(fighters_path, fights_path, out_path='model_artifacts/pairwise_model.joblib'):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.calibration import CalibratedClassifierCV

    df = pd.read_csv(fighters_path)
    fights = pd.read_csv(fights_path)
    df2, num_cols = engineer_numeric(df)

    rows, y = [], []
    for _, r in fights.iterrows():
        if r['fighter_a'] not in df2['fighter'].values or r['fighter_b'] not in df2['fighter'].values:
            continue
        va = df2[df2['fighter']==r['fighter_a']].iloc[0][num_cols].values.astype(float)
        vb = df2[df2['fighter']==r['fighter_b']].iloc[0][num_cols].values.astype(float)
        rows.append(va - vb); y.append(1)
        rows.append(vb - va); y.append(0)
    X = pd.DataFrame(rows, columns=[f'diff_{c}' for c in num_cols]).fillna(0.0)
    y = pd.Series(y, name='y')

    models = {
        'GB': GradientBoostingClassifier(random_state=123),
        'RF': RandomForestClassifier(n_estimators=800, random_state=123, n_jobs=-1),
        'LR': LogisticRegression(max_iter=2000, random_state=123)
    }
    best_name, best_auc, best_model = None, -1.0, None
    for name, m in models.items():
        proba = cross_val_predict(m, X, y, cv=5, method='predict_proba')[:,1]
        auc = roc_auc_score(y, proba)
        if auc > best_auc:
            best_name, best_auc, best_model = name, auc, m
    best_model.fit(X, y)
    cal = CalibratedClassifierCV(best_model, cv=5, method='isotonic').fit(X, y)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    joblib.dump({'model': cal, 'features': [f'diff_{c}' for c in num_cols], 'src_numeric': num_cols}, out_path)
    return best_name, float(best_auc), out_path

# Load data
if os.path.exists(fighters_path):
    fighters_df = load_csv(fighters_path)
    fighters_df, src_cols = engineer_numeric(fighters_df)
    st.success(f'Loaded fighters CSV with {fighters_df.shape[0]} rows.')
else:
    st.error('Fighters CSV not found.')

if os.path.exists(fights_path):
    fights_df = load_csv(fights_path)
    st.info(f'Loaded fights CSV with {fights_df.shape[0]} rows.')
else:
    st.warning('Fights CSV not found (you can still browse fighters).')

# Train on demand
if train_btn and os.path.exists(fighters_path) and os.path.exists(fights_path):
    name, auc, path = train_model(fighters_path, fights_path, artifacts_path)
    st.success(f'Trained {name} (CV AUC={auc:.3f}) → {path}')

# UI: choose division and fighters
if "division" in fighters_df.columns:
    divs = sorted(fighters_df["division"].astype(str).unique())
else:
    divs = ["All"]
    fighters_df["division"] = "All"

div = st.selectbox("Division", divs, index=0)
pool = fighters_df[fighters_df["division"] == div].copy()

col1, col2 = st.columns(2)
with col1:
    a_name = st.selectbox("Fighter A", sorted(pool["fighter"].unique()))

with col2:
    if pool["fighter"].nunique() < 2:
        # If only one fighter in this division, let user choose opponent from full roster
        st.info("Only one fighter in this division. Choose an opponent from the full roster.")
        other_pool = fighters_df[fighters_df["fighter"] != a_name]
        b_name = st.selectbox("Fighter B (any division)", sorted(other_pool["fighter"].unique()))
        b = other_pool[other_pool["fighter"] == b_name].iloc[0]
    else:
        b_name = st.selectbox("Fighter B", sorted([x for x in pool["fighter"].unique() if x != a_name]))
        b = pool[pool["fighter"] == b_name].iloc[0]

# Always resolve A from the current division pool
a = pool[pool["fighter"] == a_name].iloc[0]

def show_card(row):
    meta = ['age','height_cm','reach_cm','wins','losses','draws','ko_wins',
            'fights_last_24mo','days_since_last_fight','ko_rate','reach_to_height','stance']
    with st.container(border=True):
        st.markdown(f"#### {row['fighter']}  \n*{row['division']}*")
        cols = st.columns(4)
        for i, k in enumerate(meta):
            cols[i%4].metric(k.replace('_',' ').title(), row.get(k,'-'))



st.markdown('---')
st.subheader('Tale of the Tape')
show_card(a)
show_card(b)

st.markdown('---')
st.subheader('Win Probability')
if os.path.exists(artifacts_path):
    bundle = joblib.load(artifacts_path)
    model, num_cols = bundle['model'], bundle['src_numeric']
    X_ab = make_diff(a, b, num_cols)
    p_ab = float(model.predict_proba(X_ab)[0,1])
    st.success(f"P({a_name} beats {b_name}): {p_ab:.3f}")
    st.progress(p_ab, text=f"{a_name} win probability")
else:
    st.warning("No model file found. Click 'Train / Retrain model' in the sidebar after you set Fighters & Fights CSVs.")
