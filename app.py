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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

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
st.title("🩺 Diabetes Risk Predictor")
st.caption("Educational project. This is a screening estimate, not a medical diagnosis. "
           "Your inputs are not saved or stored.")

DATA_FILE = Path(__file__).with_name("data.csv")   # put your cleaned CSV here for public hosting
MAX_ROWS = 300_000          # safety limit on dataset size
MAX_CALLS, WINDOW = 15, 600  # max risk calculations per visitor per 10 minutes


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

# ------------------------------------------------------------------ main
t1, t2 = st.tabs(["Check my risk", "About the model"])

with t1:
    st.subheader("Enter your details")
    vals = {}
    c1, c2 = st.columns(2)
    for i, f in enumerate(features):
        box = c1 if i % 2 == 0 else c2
        m = info[f]
        if m["binary"]:
            vals[f] = float(box.selectbox(f, [0, 1], index=int(round(m["median"])),
                                          format_func=lambda v: "Yes" if v == 1 else "No", key=f))
        else:
            step = 0.1 if (m["max"] - m["min"]) < 60 else 1.0
            vals[f] = float(box.number_input(f, float(m["min"]), float(m["max"]),
                                              float(m["median"]), step=step, key=f))

    if st.button("Calculate risk", type="primary"):
        if call_allowed():
            # keep every value inside the range seen in the training data
            st.session_state["vals"] = {f: float(np.clip(vals[f], info[f]["min"], info[f]["max"]))
                                        for f in features}
        else:
            st.warning("Too many requests. Please wait a few minutes and try again.")

    if "vals" in st.session_state and set(st.session_state["vals"]) == set(features):
        v = st.session_state["vals"]
        x = pd.DataFrame([v])[features]
        p = float(model.predict_proba(x)[:, 1][0])
        lvl = risk_level(p, low_cut, high_cut)
        color = {"Low": "green", "Medium": "orange", "High": "red"}[lvl]

        st.markdown(f"### Risk level: :{color}[{lvl}]")
        st.metric("Estimated probability", f"{p * 100:.1f}%",
                  help="Compared with the average in the training data: "
                       f"{stats['prev'] * 100:.1f}%")
        st.progress(min(p, 1.0))

        bmi_col = next((f for f in features if "bmi" in f.lower()), None)
        if bmi_col:
            st.write(f"BMI category: **{bmi_category(v[bmi_col])}**")

        # why: replace each input by a typical value and see how much risk falls
        rows = []
        for f in features:
            alt = dict(v)
            alt[f] = round(info[f]["median"]) if info[f]["binary"] else info[f]["median"]
            q = float(model.predict_proba(pd.DataFrame([alt])[features])[:, 1][0])
            rows.append((f, p - q))
        why = pd.DataFrame(rows, columns=["Input", "Effect"]).sort_values("Effect", ascending=False)
        up_f = why[why["Effect"] > 0.001].head(5)
        st.markdown("**What is raising your risk** (compared with a typical person)")
        if len(up_f):
            st.bar_chart(up_f.set_index("Input")["Effect"] * 100)
            st.caption("Bars show percentage points of risk added by each input.")
        else:
            st.write("None of your inputs is raising your risk above typical.")

        st.markdown("**Advice**")
        for tip in ADVICE[lvl]:
            st.write("- " + tip)

        st.markdown("---")
        st.markdown("**What if I change something?**")
        wf = st.selectbox("Input to change", features, key="wf_feature")
        m = info[wf]
        if m["binary"]:
            nv = float(st.selectbox("New value", [0, 1], format_func=lambda z: "Yes" if z == 1 else "No", key="wf_bin"))
        else:
            nv = float(st.slider("New value", float(m["min"]), float(m["max"]), float(v[wf]), key="wf_num"))
        alt = dict(v); alt[wf] = nv
        p2 = float(model.predict_proba(pd.DataFrame([alt])[features])[:, 1][0])
        st.metric("New estimated risk", f"{p2 * 100:.1f}%", f"{(p2 - p) * 100:+.1f} points",
                  delta_color="inverse")

with t2:
    st.subheader("Model details")
    a, b, c = st.columns(3)
    a.metric("ROC-AUC", f"{stats['roc']:.3f}")
    b.metric("PR-AUC", f"{stats['pr']:.3f}")
    c.metric("Training rows", f"{stats['rows']:,}")
    st.write(f"Algorithm: Gradient Boosting with probability calibration. "
             f"Diabetes share in data: **{stats['prev'] * 100:.1f}%**.")
    st.write("Scores come from a held-out 20% test set.")
    st.markdown("**Limitations**")
    st.write("- Estimates current risk from survey-style inputs; it does not predict future years.\n"
             "- Accuracy depends on the dataset's quality and how well it represents Indian patients.\n"
             "- Not a diagnosis. Always confirm with a blood test and a doctor.")
