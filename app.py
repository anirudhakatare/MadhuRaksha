"""
MadhuRaksha - diabetes risk checker (Streamlit)
Design: "Clinical Sanctuary" (light + dark). Model: Random Forest trained on the Pima Indians dataset (768 real patients).
Research prototype. NOT a medical diagnosis.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="MadhuRaksha", page_icon="🩺", layout="wide", initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------------------------
# 1. MODEL  (trained once when the app starts; no pickle files, so no version problems on the cloud)
# ----------------------------------------------------------------------------------------------
COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
        "DiabetesPedigreeFunction", "Age", "Outcome"]
FEATURES = COLS[:-1]
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


@st.cache_resource(show_spinner="Preparing the model...")
def load_model():
    here = os.path.dirname(os.path.abspath(__file__))
    path = next((os.path.join(here, p) for p in ["diabetes.csv", "pima-indians-diabetes.csv"]
                 if os.path.exists(os.path.join(here, p))), None)
    if path is None:
        st.error("Data file not found. Put diabetes.csv (Pima dataset) next to app.py.")
        st.stop()
    first = open(path, encoding="utf-8", errors="ignore").readline().split(",")[0].strip()
    has_header = not first.replace(".", "", 1).isdigit()
    df = pd.read_csv(path) if has_header else pd.read_csv(path, header=None, names=COLS)
    df[ZERO_MEANS_MISSING] = df[ZERO_MEANS_MISSING].replace(0, np.nan)
    X, y = df[FEATURES], df["Outcome"]
    pipe = make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1))
    oof = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=7), method="predict_proba")[:, 1]
    _, rec, thr = precision_recall_curve(y, oof)
    cutoff = float(thr[np.where(rec[:-1] >= 0.85)[0].max()])      # catches ~85% of diabetic patients
    pipe.fit(X, y)
    return pipe, cutoff, len(df)


MODEL, CUTOFF, N_TRAIN = load_model()


def risk(glucose, diastolic, bmi, age, pedigree):
    """Probability of diabetes. Inputs the form does not ask for (pregnancies, skin fold, insulin) are filled by the model."""
    row = pd.DataFrame([[np.nan, glucose, diastolic, np.nan, np.nan, bmi, pedigree, age]], columns=FEATURES)
    return float(MODEL.predict_proba(row)[0][1])


# ----------------------------------------------------------------------------------------------
# 2. THEME (light / dark) - colours from the Clinical Sanctuary design files
# ----------------------------------------------------------------------------------------------
LIGHT = dict(page="#F4F6F9", card="#FFFFFF", border="#E2E8F0", text="#0F172A", muted="#64748B", field="#F8FAFC",
             accent="#0D9488", accent_soft="#CCFBF1", panel="linear-gradient(165deg,#1E1B4B 0%,#0F172A 100%)",
             panel_border="rgba(255,255,255,0.08)", header="rgba(255,255,255,0.85)")
DARK = dict(page="#090D16", card="#0F172A", border="#1E293B", text="#F1F5F9", muted="#94A3B8", field="#182337",
            accent="#14B8A6", accent_soft="#134E4A", panel="linear-gradient(180deg,#0F172A 0%,#111A30 60%,#0A0F1D 100%)",
            panel_border="#1E293B", header="rgba(9,13,22,0.88)")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root{--page:__page__;--card:__card__;--border:__border__;--text:__text__;--muted:__muted__;--field:__field__;--accent:__accent__;--accent-soft:__accent_soft__;}
html,body,.stApp,[data-testid="stAppViewContainer"]{background:var(--page)!important;color:var(--text);font-family:'Plus Jakarta Sans',-apple-system,'Segoe UI',sans-serif;}
[data-testid="stHeader"]{background:transparent!important;} #MainMenu,footer{visibility:hidden;}
.block-container{max-width:1200px;padding:1rem 1rem 3rem;}
h1,h2,h3,.serif{font-family:'Newsreader',Georgia,serif!important;color:var(--text)!important;font-weight:500!important;}
[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label,.stRadio label,.stToggle label p{color:var(--muted)!important;font-weight:600;font-size:13px;}
.st-key-inputs{background:var(--card);border:1px solid var(--border)!important;border-radius:24px;padding:1.25rem 1.25rem .5rem;box-shadow:0 10px 30px -5px rgba(15,23,42,.06);}
[data-testid="stNumberInput"] input,[data-baseweb="input"],[data-baseweb="base-input"]{background:var(--field)!important;color:var(--text)!important;border-color:var(--border)!important;border-radius:12px!important;}
[data-baseweb="input"]{border:1px solid var(--border)!important;}[data-baseweb="input"]:focus-within{border-color:var(--accent)!important;}
[data-testid="stNumberInput"] button{background:var(--field)!important;color:var(--text)!important;}
[data-testid="stSlider"] [data-testid="stThumbValue"],[data-testid="stSliderTickBarMin"],[data-testid="stSliderTickBarMax"]{color:var(--text)!important;font-weight:700;}
button[data-variant="segmented_control"],button[data-testid="stBaseButton-segmented_control"]{background:var(--card)!important;border:1px solid var(--border)!important;color:var(--muted)!important;border-radius:12px!important;font-weight:600;}
button[data-variant="segmented_control"][aria-checked="true"],button[data-variant="segmented_control"][data-selected="true"],button[data-testid="stBaseButton-segmented_controlActive"]{background:var(--accent)!important;border-color:var(--accent)!important;color:#fff!important;font-weight:700;}
button[data-variant="segmented_control"][aria-checked="true"] *,button[data-testid="stBaseButton-segmented_controlActive"] *{color:#fff!important;}
[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important;}
.st-key-sim{background:__panel__;border:1px solid __panel_border__!important;border-radius:24px;padding:1.1rem 1.25rem .75rem;margin-top:1rem;}
.st-key-sim [data-testid="stWidgetLabel"] p{color:#CBD5E1!important;} .st-key-sim [data-testid*="Thumb"],.st-key-sim [data-testid*="TickBar"]{color:#F1F5F9!important;}
.hdr{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:.25rem 0 .75rem;}
.brand{display:flex;align-items:center;gap:12px;} .logo{width:42px;height:42px;border-radius:50%;background:#0F172A;display:flex;align-items:center;justify-content:center;border:1px solid var(--border);}
.brand h1{font-size:26px!important;margin:0!important;line-height:1.1!important;padding:0!important;} .brand p{margin:0;color:var(--muted);font-size:12px;}
.pill{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:9999px;background:var(--card);border:1px solid var(--border);font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);}
.dot{width:6px;height:6px;border-radius:50%;display:inline-block;}
.sec{font-family:'Newsreader',Georgia,serif;font-size:22px;color:var(--text);margin:.25rem 0 .1rem;} .sub{color:var(--muted);font-size:12.5px;margin-bottom:.75rem;}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin:-.25rem 0 .9rem;} .chip{font-size:11px;font-weight:700;padding:3px 10px;border-radius:9999px;border:1px solid var(--border);color:var(--muted);background:var(--card);}
.chip.on{background:var(--accent);border-color:var(--accent);color:#fff;}
.note{font-size:11.5px;color:var(--muted);line-height:1.5;margin:-.4rem 0 .9rem;}
.panel{position:relative;overflow:hidden;border-radius:24px;padding:22px 20px;color:#fff;background:__panel__;border:1px solid __panel_border__;box-shadow:0 20px 40px -12px rgba(15,23,42,.35);}
.panel svg.mandala{position:absolute;right:-60px;bottom:-60px;width:250px;height:250px;opacity:.07;color:#2DD4BF;pointer-events:none;}
.tag{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:9999px;background:rgba(20,184,166,.15);border:1px solid rgba(20,184,166,.3);color:#5EEAD4;font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;}
.gauge{position:relative;width:176px;height:176px;margin:14px auto 6px;} .gauge .num{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.gauge .big{font-size:44px;font-weight:700;letter-spacing:-.03em;line-height:1;} .gauge .cap{font-size:10.5px;color:#94A3B8;text-transform:uppercase;letter-spacing:.08em;margin-top:4px;}
.badge{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:9999px;font-size:11px;font-weight:700;border:1px solid;}
.head{font-family:'Newsreader',Georgia,serif;font-size:27px;line-height:1.2;margin:10px 0 4px;color:#fff;} .lead{color:#CBD5E1;font-size:13px;line-height:1.55;max-width:420px;margin:0 auto;}
.cta{margin:14px 0 0;padding:10px 12px;border-radius:14px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);font-size:12.5px;color:#E2E8F0;line-height:1.5;}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:16px;} @media(max-width:640px){.grid{grid-template-columns:1fr;}}
.card{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:12px 14px;backdrop-filter:blur(12px);}
.card .t{font-size:11px;color:#94A3B8;font-weight:700;letter-spacing:.06em;text-transform:uppercase;} .card .v{font-size:20px;font-weight:700;margin:2px 0;color:#fff;}
.card .d{font-size:11.5px;color:#94A3B8;line-height:1.45;margin-top:4px;} .row{display:flex;align-items:center;justify-content:space-between;gap:8px;}
.mini{font-size:10px;font-weight:700;padding:2px 8px;border-radius:9999px;border:1px solid;} .pips{display:flex;gap:4px;margin-top:8px;} .pip{height:5px;flex:1;border-radius:9999px;background:rgba(255,255,255,.14);} .pip.on{background:#F59E0B;box-shadow:0 0 6px #F59E0B;}
.sech{font-family:'Newsreader',Georgia,serif;font-size:19px;margin:20px 0 8px;color:#fff;}
.act{display:flex;gap:10px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-radius:14px;padding:11px 13px;margin-bottom:8px;}
.act i{width:8px;height:8px;border-radius:50%;background:#F59E0B;margin-top:6px;flex:0 0 8px;box-shadow:0 0 6px #F59E0B;} .act.med i{background:#14B8A6;box-shadow:0 0 6px #14B8A6;}
.act b{display:block;font-size:13.5px;color:#fff;margin-bottom:2px;} .act span{font-size:12px;color:#94A3B8;line-height:1.5;}
.simres{display:flex;align-items:baseline;gap:10px;color:#fff;margin:2px 0 8px;} .simres .n{font-size:30px;font-weight:700;letter-spacing:-.02em;} .simres .l{font-size:12px;color:#CBD5E1;}
.foot{font-size:11px;color:var(--muted);line-height:1.6;margin-top:18px;border-top:1px solid var(--border);padding-top:12px;}
"""


def inject_css(dark):
    t = DARK if dark else LIGHT
    css = CSS
    for k, v in t.items():
        css = css.replace(f"__{k}__", v)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def H(s):
    """Collapse HTML to one line: Markdown would treat blank lines or indents as code blocks."""
    return "".join(line.strip() for line in s.strip().splitlines())


# ----------------------------------------------------------------------------------------------
# 3. HEADER + THEME TOGGLE
# ----------------------------------------------------------------------------------------------
if "dark" not in st.session_state:
    st.session_state.dark = False
dark = st.session_state.dark
inject_css(dark)

h1, h2, h3 = st.columns([6, 2.2, 1.8], vertical_alignment="center")
with h1:
    st.markdown(H("""
    <div class="hdr"><div class="brand">
      <div class="logo"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2DD4BF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.5C12 2.5 5 10 5 14.5a7 7 0 0014 0C19 10 12 2.5 12 2.5z"/><path d="M9.5 14.5l1.8 1.8 3.4-3.6"/></svg></div>
      <div><h1>MadhuRaksha</h1><p>Know your risk. Protect your future.</p></div></div></div>"""), unsafe_allow_html=True)
with h2:
    st.markdown(H("""<div style="text-align:right"><span class="pill"><span class="dot" style="background:#FF9933"></span><span class="dot" style="background:#fff;border:1px solid #CBD5E1"></span><span class="dot" style="background:#138808"></span>&nbsp;Made for India</span></div>"""), unsafe_allow_html=True)
with h3:
    st.toggle("Dark mode", key="dark")

# ----------------------------------------------------------------------------------------------
# 4. INPUTS (left) and RESULTS (right)
# ----------------------------------------------------------------------------------------------
FAMILY = {"None": 0.25, "One parent": 0.50, "Sibling": 0.60, "Both parents": 0.90}   # approximate pedigree scores


def seg(label, options, default, key):
    v = st.segmented_control(label, options, default=default, key=key)
    return v if v else default


left, right = st.columns([5, 6], gap="large")

with left:
    with st.container(border=True, key="inputs"):
        st.markdown('<div class="sec">Your health snapshot</div><div class="sub">Move the sliders. Results update instantly.</div>',
                    unsafe_allow_html=True)
        sex = seg("Sex", ["Female", "Male"], "Female", "sex")
        age = st.slider("Age (years)", 21, 81, 45)

        c1, c2 = st.columns(2)
        height = c1.number_input("Height (cm)", 120, 220, 165)
        weight = c2.number_input("Weight (kg)", 30, 200, 75)
        bmi = round(weight / (height / 100) ** 2, 1)
        tiers = [("Lean", bmi < 23), ("Borderline", 23 <= bmi < 25), ("Elevated", 25 <= bmi < 30), ("Obese", bmi >= 30)]
        st.markdown(f'<div class="sub" style="margin:0 0 4px">Your BMI: <b style="color:var(--text);font-size:15px">{bmi}</b></div>'
                    + '<div class="chips">' + "".join(f'<span class="chip{" on" if on else ""}">{n}</span>' for n, on in tiers) + "</div>"
                    '<div class="note">India-specific cut-offs: health risk starts at BMI 23, higher than the global 25.</div>',
                    unsafe_allow_html=True)

        waist = st.slider("Waist (inches)", 24, 60, 34)
        glucose = st.slider("Blood glucose, 2 hours after a glucose drink (mg/dL)", 60, 250, 130)
        st.markdown('<div class="note">The model learned from this kind of test (OGTT). A fasting value is lower, so it would '
                    'under-estimate risk here.</div>', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        sbp = b1.number_input("BP upper / systolic", 80, 220, 125)
        dbp = b2.number_input("BP lower / diastolic", 40, 140, 80)

        diet = seg("Usual diet", ["Rice-heavy", "Roti-based", "Millet-based", "Sweets often"], "Rice-heavy", "diet")
        family = seg("Family history of diabetes", list(FAMILY), "None", "family")
        activity = seg("Physical activity", ["Low", "Medium", "High"], "Low", "activity")
        st.markdown('<div class="note">Diet, activity and waist do not change the score (the training data does not have them). '
                    'They shape your advice.</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------
# 5. CALCULATIONS
# ----------------------------------------------------------------------------------------------
pedigree = FAMILY[family]
p = risk(glucose, dbp, bmi, age, pedigree)
REF = {"glucose": 100, "bmi": 22.0, "dbp": 75, "ped": 0.25}              # "healthy reference" values for what-if comparison
d_glu = max(0.0, p - risk(REF["glucose"], dbp, bmi, age, pedigree))
d_bmi = max(0.0, p - risk(glucose, dbp, REF["bmi"], age, pedigree))
d_bp = max(0.0, p - risk(glucose, REF["dbp"], bmi, age, pedigree))
d_fam = max(0.0, p - risk(glucose, dbp, bmi, age, REF["ped"]))

if p < 0.20:
    tier, tcol, grad, headline = "Low", "#2DD4BF", ("#14B8A6", "#5EEAD4"), "Your estimated risk is low"
elif p < 0.45:
    tier, tcol, grad, headline = "Moderate", "#FBBF24", ("#F59E0B", "#FBBF24"), "Your estimated risk is moderate"
else:
    tier, tcol, grad, headline = "High", "#FB923C", ("#EA580C", "#F59E0B"), "Your estimated risk is high"

CIRC = 2 * np.pi * 48
offset = CIRC * (1 - max(p, 0.02))
pct, in100 = round(p * 100), round(p * 100)
flagged = p >= CUTOFF


def status(kind):
    if kind == "glucose":
        if glucose >= 200: return "Diabetes range", 5
        if glucose >= 140: return "Prediabetes range", 4 if glucose >= 170 else 3
        return "Normal range", 2 if glucose >= 120 else 1
    if kind == "bmi":
        if bmi >= 30: return "Obese", 5
        if bmi >= 25: return "Elevated", 4 if bmi >= 27.5 else 3
        if bmi >= 23: return "Borderline", 2
        return "Lean", 1
    if kind == "bp":
        if sbp >= 140 or dbp >= 90: return "High", 5
        if sbp >= 130 or dbp >= 80: return "Raised", 3
        if sbp >= 120: return "Elevated", 2
        return "Normal", 1


def pips(n):
    return '<div class="pips">' + "".join(f'<div class="pip{" on" if i < n else ""}"></div>' for i in range(5)) + "</div>"


def driver(title, value, badge, level, delta, desc, note=None):
    col = "#FBBF24" if level >= 3 else "#5EEAD4"
    adds = delta * 100 >= 0.5
    dtxt = f"+{delta * 100:.0f} pts" if adds else (note or "no added risk")
    dcol = col if adds else "#94A3B8"
    return (f'<div class="card"><div class="row"><span class="t">{title}</span>'
            f'<span class="mini" style="color:{col};border-color:{col}55;background:{col}1A">{badge}</span></div>'
            f'<div class="row"><span class="v">{value}</span><span style="font-size:12px;font-weight:700;color:{dcol}">{dtxt}</span></div>'
            f'{pips(level)}<div class="d">{desc}</div></div>')


g_badge, g_lvl = status("glucose")
b_badge, b_lvl = status("bmi")
p_badge, p_lvl = status("bp")
act_lvl = {"Low": 4, "Medium": 2, "High": 1}[activity]
act_txt = {"Low": "Under 150 min/week. Building up is the single best habit change.",
           "Medium": "A good base. Aim for 150+ min/week and add 2 strength days.",
           "High": "Excellent. Keep it up."}[activity]
fam_lvl = {"None": 1, "One parent": 3, "Sibling": 3, "Both parents": 5}[family]

drivers_html = (
    driver("Glucose", f"{glucose} mg/dL", g_badge, g_lvl, d_glu, "Compared with a healthy 100 mg/dL after the glucose drink.")
    + driver("BMI", f"{bmi}", b_badge, b_lvl, d_bmi, "Compared with a BMI of 22. Using India-specific cut-offs.")
    + driver("Blood pressure", f"{sbp}/{dbp}", p_badge, p_lvl, d_bp, "Lower number compared with 75 mmHg.")
    + driver("Family history", family, "Genetic", fam_lvl, d_fam, "Approximate: the dataset scores how many relatives had diabetes.")
    + driver("Activity", activity, "Advice only", act_lvl, 0.0, act_txt, note="not in score")
)

# ---- advice rules (guideline-style thresholds) ----
waist_cut = 35.5 if sex == "Male" else 31.5                             # 90 cm men, 80 cm women (Asian Indian cut-offs)
prev = []
if bmi >= 23:
    prev.append(("Move towards a healthier weight", "Aim to lose 5-7% of your body weight gradually. Asian Indians are at higher risk at lower BMI, so even a few kilos help."))
if waist >= waist_cut:
    prev.append(("Reduce belly fat", f"Your waist ({waist} in) is above the {waist_cut} in limit for {sex.lower()}s. Belly fat raises insulin resistance more than weight alone."))
if activity != "High":
    prev.append(("Build up exercise", "Work towards 150 minutes a week of brisk walking, cycling or similar (30 minutes, 5 days), plus 2 days of strength work."))
prev.append({"Rice-heavy": ("Shift your plate", "Replace part of polished white rice with millets (ragi, jowar, bajra) or brown rice, and add dal and vegetables to fill half the plate."),
             "Roti-based": ("Upgrade your rotis", "Choose whole-wheat, multigrain or millet rotis, and pair them with dal, vegetables and curd rather than extra rice or fried sides."),
             "Millet-based": ("Keep your millets", "A millet-based diet is a good base. Watch fried snacks and sugary drinks."),
             "Sweets often": ("Cut sugar spikes", "Keep mithai, sugary chai and cold drinks for occasional treats and choose fruit or nuts as snacks.")}[diet])
if sbp >= 130 or dbp >= 80:
    prev.append(("Lower salt", "Cut down on pickles, papad, namkeen and added salt, and get your blood pressure re-checked within a few weeks."))
if family != "None":
    prev.append(("Screen more often", "Because of your family history, test your blood sugar every year even if results look fine."))

med = []
if glucose >= 200:
    med.append(("See a doctor soon", "Your glucose is in the diabetes range on this test. A doctor must confirm it with repeat tests (fasting glucose and HbA1c) and discuss treatment."))
elif glucose >= 140:
    med.append(("See a doctor in the next few weeks", "Your glucose is in the prediabetes range. Ask for HbA1c and fasting glucose tests and about a diabetes-prevention programme."))
elif flagged:
    med.append(("Get a blood sugar test", "Your profile is above the screening cut-off, so ask for fasting glucose and HbA1c tests."))
else:
    med.append(("Routine check-up", "Test your blood sugar every 1 to 3 years, or yearly if you have risk factors."))
if sbp >= 140 or dbp >= 90:
    med.append(("Check your blood pressure", "Your reading is clearly high (140/90 or above). Please have a doctor check it."))

act_html = "".join(f'<div class="act"><i></i><div><b>{a}</b><span>{b}</span></div></div>' for a, b in prev[:5])
med_html = "".join(f'<div class="act med"><i></i><div><b>{a}</b><span>{b}</span></div></div>' for a, b in med)

lead = f"In the real patient data this model learned from, about {in100} in 100 people with a profile like yours had diabetes."
cta = ("Your profile is above the screening cut-off, so a proper blood sugar test is advised."
       if flagged else "Your profile is below the screening cut-off. Keep up healthy habits and check regularly.")

with right:
    st.markdown(H(f"""
    <div class="panel">
      <svg class="mandala" fill="none" stroke="currentColor" viewBox="0 0 100 100"><circle cx="50" cy="50" r="45" stroke-dasharray="2 2" stroke-width="0.8"/><circle cx="50" cy="50" r="35" stroke-width="0.6"/><circle cx="50" cy="50" r="25" stroke-width="0.5"/><path d="M50 5 Q65 50 50 95 Q35 50 50 5" stroke-width="0.6"/><path d="M5 50 Q50 65 95 50 Q50 35 5 50" stroke-width="0.6"/><path d="M18 18 Q50 50 82 82 Q50 50 18 18" stroke-width="0.6"/><path d="M82 18 Q50 50 18 82 Q50 50 82 18" stroke-width="0.6"/></svg>
      <div style="text-align:center;position:relative">
        <span class="tag"><span class="dot" style="background:#2DD4BF"></span>Dynamic clinical assessment</span>
        <div class="gauge">
          <svg viewBox="0 0 120 120" width="176" height="176" style="transform:rotate(-90deg)">
            <defs><linearGradient id="rg" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="{grad[0]}"/><stop offset="100%" stop-color="{grad[1]}"/></linearGradient></defs>
            <circle cx="60" cy="60" r="48" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="10"/>
            <circle cx="60" cy="60" r="48" fill="none" stroke="url(#rg)" stroke-width="11" stroke-linecap="round" stroke-dasharray="{CIRC:.1f}" stroke-dashoffset="{offset:.1f}"/>
          </svg>
          <div class="num"><span class="big">{pct}%</span><span class="cap">Estimated risk</span></div>
        </div>
        <span class="badge" style="color:{tcol};border-color:{tcol}66;background:{tcol}1F"><span class="dot" style="background:{tcol}"></span>{tier} risk</span>
        <div class="head">{headline}</div>
        <p class="lead">{lead}</p>
        <div class="cta">{cta}</div>
      </div>
      <div class="sech">What is driving your result</div>
      <div class="grid">{drivers_html}</div>
      <div class="sech">Your action plan</div>
      {act_html}
      <div class="sech">Medical follow-up</div>
      {med_html}
    </div>"""), unsafe_allow_html=True)

    # ---- what-if simulator (uses the real model) ----
    with st.container(key="sim"):
        st.markdown('<div class="sech" style="margin-top:0">What if you improved?</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        sg = s1.slider("Glucose", 60, 250, int(glucose), key=f"sg_{glucose}")
        sb = s2.slider("BMI", 15.0, 45.0, float(bmi), 0.5, key=f"sb_{bmi}")
        sd = s3.slider("BP lower", 40, 140, int(dbp), key=f"sd_{dbp}")
        p2 = risk(sg, sd, sb, age, pedigree)
        diff = (p2 - p) * 100
        col = "#5EEAD4" if diff < -0.5 else ("#FBBF24" if diff > 0.5 else "#CBD5E1")
        word = "lower than now" if diff < -0.5 else ("higher than now" if diff > 0.5 else "the same as now")
        st.markdown(H(f"""<div class="simres"><span class="n">{round(p2 * 100)}%</span>
            <span class="l" style="color:{col}">{abs(diff):.0f} points {word} ({pct}%)</span></div>"""), unsafe_allow_html=True)

st.markdown(H(f"""<div class="foot"><b>Please read.</b> MadhuRaksha is a research prototype for a college project, not a medical device, and it cannot diagnose diabetes.
Only a doctor can, using blood tests. The model is a Random Forest trained on the Pima Indians dataset ({N_TRAIN} women aged 21 and above), so it may not
fit every person or group in India. Glucose is the 2-hour test used in that dataset. Family history is turned into the dataset's pedigree score by an approximation.
Pregnancies, skin-fold thickness and insulin are not asked, so typical values from the data are used. The score shows how similar people in that data fared today. It is not a
10-year forecast.</div>"""), unsafe_allow_html=True)
