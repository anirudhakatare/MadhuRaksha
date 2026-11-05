"""Diabetes Risk Predictor - college R&D project.
Run:  streamlit run app.py
Upload your cleaned CSV in the sidebar (or use the built-in demo data).
"""
import hmac
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="MadhuRaksha - Diabetes Risk", page_icon="🩺", layout="centered")
st.markdown("""<style>
.stApp{background:#090D16}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.2rem;max-width:520px}
[data-testid="stExpander"]{background:#131C2E;border:1px solid #1E293B;border-radius:16px}
.stApp, .stApp p, .stApp li, .stApp label, .stApp span{color:#E2E8F0}
iframe{border:0}
</style>""", unsafe_allow_html=True)

ui = components.declare_component("madhuraksha_ui", path=str(Path(__file__).parent / "ui"))

ADVICE = {
    "Low": [
        "Keep a healthy weight (for Indians, BMI under 23 is the safer range).",
        "Stay active: about 150 minutes of brisk walking or similar per week.",
        "Prefer whole grains, dal, vegetables; limit sugary drinks and sweets.",
        "Get a blood sugar check about every 3 years, or yearly after age 35.",
    ],
    "Medium": [
        "Get an HbA1c or fasting blood sugar test in the next few months.",
        "Aim to lose 5-7% of body weight if overweight.",
        "Cut refined carbs (white rice, maida, sweets) and fried food.",
        "Add 30 minutes of daily activity and reduce long sitting hours.",
    ],
    "High": [
        "Please see a doctor soon for HbA1c, fasting glucose and a full check-up.",
        "Do not wait for symptoms; early diabetes often has none.",
        "Start diet and activity changes now, ideally with professional guidance.",
        "Check blood pressure and cholesterol too, as risks often come together.",
    ],
}


@st.cache_data
def demo_data(n=20000, seed=7):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n)
    bmi = np.clip(rng.normal(25.5, 4.5, n), 15, 50)
    glucose = np.clip(rng.normal(100, 22, n) + (bmi - 25) * 0.8, 60, 300)
    bp = np.clip(rng.normal(120, 15, n) + (age - 40) * 0.3, 80, 200)
    fam = rng.binomial(1, 0.25, n)
    active = rng.binomial(1, 0.45, n)
    z = (-7.2 + 0.035 * age + 0.11 * (bmi - 20) + 0.045 * (glucose - 70)
         + 0.012 * (bp - 100) + 0.7 * fam - 0.5 * active)
    y = rng.binomial(1, 1 / (1 + np.exp(-z)))
    return pd.DataFrame({"Age": age, "BMI": bmi.round(1), "Glucose": glucose.round(0),
                         "BloodPressure": bp.round(0), "FamilyHistory": fam,
                         "PhysicalActivity": active, "Diabetes": y})


@st.cache_resource(show_spinner="Training model...")
def train(df, target, features, positive):
    d = df[features + [target]].dropna(subset=[target]).copy()
    if len(d) > 200_000:
        d = d.sample(200_000, random_state=0)
    y = (d[target].astype(str) == str(positive)).astype(int)
    X = d[features].astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    method = "isotonic" if len(X_tr) > 5000 else "sigmoid"
    model = CalibratedClassifierCV(HistGradientBoostingClassifier(random_state=0),
                                   method=method, cv=3).fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    stats = {"roc": roc_auc_score(y_te, p), "pr": average_precision_score(y_te, p),
             "prev": float(y.mean()), "rows": len(d)}
    info = {}
    for f in features:
        col = X[f]
        info[f] = {"min": float(col.min()), "max": float(col.max()),
                   "median": float(col.median()), "binary": set(col.dropna().unique()) <= {0.0, 1.0}}
    return model, stats, info


def risk_level(p, low_cut, high_cut):
    return "Low" if p < low_cut else ("Medium" if p < high_cut else "High")


def bmi_category(b):
    if b < 18.5: return "Underweight"
    if b < 23: return "Normal"
    if b < 25: return "Overweight (Asian cut-off)"
    return "Obese (Asian cut-off)"


# ------------------------------------------------------------------ sidebar
DATA_FILE = Path(__file__).with_name("data.csv")   # put your cleaned CSV here for public hosting
MAX_ROWS = 300_000          # safety limit on dataset size
MAX_CALLS, WINDOW = 400, 600  # live sliders recalculate often, so this is per 10 minutes


def admin_password():
    """Set ADMIN_PASSWORD in Streamlit Cloud > Settings > Secrets (never in GitHub)."""
    try:
        return st.secrets.get("ADMIN_PASSWORD")
    except Exception:
        return None


def is_admin():
    pw = admin_password()
    if not pw:
        return False                       # no password set -> app is public-only
    if st.session_state.get("is_admin"):
        return True
    with st.sidebar.expander("Admin login"):
        entered = st.text_input("Password", type="password", key="admin_pw")
        if entered and hmac.compare_digest(entered.encode(), str(pw).encode()):
            st.session_state["is_admin"] = True
            st.rerun()
        elif entered:
            st.error("Wrong password.")
    return False


def call_allowed():
    """Simple per-visitor rate limit."""
    now = time.time()
    hist = [t for t in st.session_state.get("calls", []) if now - t < WINDOW]
    ok = len(hist) < MAX_CALLS
    if ok:
        hist.append(now)
    st.session_state["calls"] = hist
    return ok


admin = is_admin()
up = None
if admin:
    with st.sidebar.expander("Data settings (admin)"):
        up = st.file_uploader("Use a different CSV (max 20 MB)", type="csv")

try:
    if up is not None:
        df = pd.read_csv(up, nrows=MAX_ROWS)
    elif DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE, nrows=MAX_ROWS)
    else:
        df = demo_data()
        if admin:
            st.sidebar.info("Using synthetic DEMO data. Add data.csv next to app.py.")
except Exception:
    st.error("The dataset could not be loaded.")
    st.stop()

cols = list(df.columns)


def guess_target(d):
    """Pick the 0/1 diabetes column (not a feature like DiabetesPedigreeFunction)."""
    names = {c.lower().replace(" ", "").replace("_", ""): c for c in d.columns}
    for key in ("outcome", "diabetes", "diabetic", "hasdiabetes", "diabetesbinary", "target", "label", "class"):
        if key in names and d[names[key]].nunique(dropna=True) == 2:
            return names[key]
    for c in d.columns:
        if d[c].nunique(dropna=True) == 2 and ("diab" in c.lower() or "outcome" in c.lower()):
            return c
    return d.columns[-1]


target = guess_target(df)
uniq = [str(u) for u in df[target].dropna().unique()][:20]
positive = "1" if "1" in uniq else (uniq[-1] if uniq else "1")
features = [c for c in df.select_dtypes("number").columns if c != target]

if admin:
    with st.sidebar.expander("Model settings (admin)"):
        target = st.selectbox("Target column", cols, index=cols.index(target))
        uniq = [str(u) for u in df[target].dropna().unique()][:20]
        positive = st.selectbox("Value meaning 'has diabetes'", uniq,
                                index=uniq.index(positive) if positive in uniq else 0)
        numeric = [c for c in df.select_dtypes("number").columns if c != target]
        features = st.multiselect("Input features", numeric, default=numeric)

if not features:
    st.warning("Choose at least one input feature in the sidebar.")
    st.stop()

model, stats, info = train(df, target, features, positive)

prev = stats["prev"]
low_cut = round(min(prev * 0.8, 0.3), 2)
high_cut = round(min(max(prev * 2.0, low_cut + 0.1), 0.6), 2)
if admin:
    with st.sidebar.expander("Risk levels (admin)"):
        low_cut, high_cut = st.slider("Low below / High above", 0.0, 1.0, (low_cut, high_cut), 0.01)

# ------------------------------------------------------------------ UI <-> model bridge
# The new UI collects these 7 inputs. Each one is matched to a dataset column by name.
TOKENS = {"age": ("age",), "bmi": ("bmi", "bodymass"), "waist": ("waist",),
          "glucose": ("gluc",), "bp": ("bloodpressure", "systolic", "sysbp", "bp"),
          "family": ("famil",), "activity": ("physicalact", "activ", "exercise")}
BINARY_ONLY = {"family", "activity"}   # only mapped when the dataset column is 0/1
REF = {"glucose": 90, "bmi": 22.0, "bp": 115, "family": "none", "activity": "high"}  # healthy reference


def map_inputs(features, info):
    """{ui_key: dataset_column} for every UI input the dataset can actually use."""
    used, mapping = set(), {}
    for key, toks in TOKENS.items():
        for f in features:
            n = f.lower().replace(" ", "").replace("_", "")
            hit = n == key or any(t in n for t in toks if len(t) > 2)
            if hit and f not in used and (key not in BINARY_ONLY or info[f]["binary"]):
                if key == "age" and not (n.startswith("age") or n.endswith("age")):
                    continue
                mapping[key] = f
                used.add(f)
                break
    return mapping


def build_row(ui_vals, mapping, features, info):
    row = {f: info[f]["median"] for f in features}
    for key, f in mapping.items():
        v = ui_vals[key]
        if key == "family":
            v = 0.0 if v == "none" else 1.0
        elif key == "activity":
            v = 0.0 if v == "low" else 1.0
        elif key == "waist" and info[f]["max"] > 80:   # dataset stores cm
            v = float(v) * 2.54
        row[f] = float(np.clip(float(v), info[f]["min"], info[f]["max"]))
    return row


def compute_result(payload, model, info, features, prev, low_cut, high_cut):
    mapping = map_inputs(features, info)
    cur, scn = payload["st"], payload["sc"]
    rows, names = [build_row(cur, mapping, features, info), build_row(scn, mapping, features, info)], ["cur", "scn"]
    for key, ref in REF.items():
        alt = dict(cur)
        alt[key] = ref
        if key == "bmi":
            alt["waist"] = 31
        rows.append(build_row(alt, mapping, features, info))
        names.append(key)
    p = model.predict_proba(pd.DataFrame(rows)[features])[:, 1] * 100
    base = dict(zip(names, p))
    drivers = {k: (round(base["cur"] - base[k], 2) if (k in mapping or (k == "bmi" and "waist" in mapping)) else 0.0)
               for k in REF}
    lvl = risk_level(base["cur"] / 100, low_cut, high_cut)
    return {"seq": payload["seq"], "risk": float(base["cur"]), "scRisk": float(base["scn"]),
            "level": lvl, "drivers": drivers, "prev": prev * 100,
            "unused": [k for k in TOKENS if k not in mapping], "advice": ADVICE[lvl]}


# ------------------------------------------------------------------ main
payload = ui(result=st.session_state.get("ui_result"), key="madhuraksha", default=None)
if payload and payload.get("seq") != (st.session_state.get("ui_result") or {}).get("seq"):
    if call_allowed():
        st.session_state["ui_result"] = compute_result(payload, model, info, features, prev, low_cut, high_cut)
        st.rerun()
    else:
        st.warning("Too many requests. Please wait a few minutes and try again.")

with st.expander("About the model"):
    a, b, c = st.columns(3)
    a.metric("ROC-AUC", f"{stats['roc']:.3f}")
    b.metric("PR-AUC", f"{stats['pr']:.3f}")
    c.metric("Training rows", f"{stats['rows']:,}")
    st.write(f"Algorithm: Gradient Boosting with probability calibration. "
             f"Diabetes share in data: **{stats['prev'] * 100:.1f}%**.")
    st.write("Scores come from a held-out 20% test set.")
    used = map_inputs(features, info)
    st.write("Inputs the model uses: " + (", ".join(f"{k} → `{v}`" for k, v in used.items()) or "none matched")
             + ". Inputs not in your dataset (for example waist or diet) are shown in the UI but do not change the risk.")
    st.markdown("**Limitations**")
    st.write("- Estimates current risk from survey-style inputs; it does not predict future years.\n"
             "- Accuracy depends on the dataset's quality and how well it represents Indian patients.\n"
             "- Not a diagnosis. Always confirm with a blood test and a doctor.")
