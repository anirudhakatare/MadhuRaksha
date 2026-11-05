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

import tempfile

UI_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#0B0F19;--card:#131C2E;--inner:#182337;--deep:#0F172A;--line:#1E293B;--line2:#334155;
--teal:#14B8A6;--tealD:#0D9488;--amber:#F59E0B;--rose:#F43F5E;--t1:#F1F5F9;--t2:#CBD5E1;--t3:#94A3B8;--t4:#64748B}
*{box-sizing:border-box;margin:0;padding:0}
body{background:transparent;color:var(--t1);font-family:'Plus Jakarta Sans',-apple-system,'Segoe UI',sans-serif;font-size:14px;-webkit-font-smoothing:antialiased}
.serif{font-family:'Newsreader',Georgia,serif}
#app{max-width:460px;margin:0 auto;display:flex;flex-direction:column;gap:16px;padding:4px 2px 20px}
header{display:flex;justify-content:space-between;align-items:center;padding:6px 2px}
.brand{display:flex;gap:10px;align-items:center}
.logo{width:40px;height:40px;border-radius:50%;background:var(--deep);border:1px solid rgba(20,184,166,.4);display:grid;place-items:center}
h1{font-family:'Newsreader',serif;font-weight:600;font-size:21px;line-height:1.1}
.sub{font-size:11px;color:var(--t3)}
.pill{display:inline-flex;align-items:center;gap:6px;font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:4px 10px;border-radius:99px;background:var(--deep);border:1px solid var(--line);color:var(--t2)}
.dots i{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:2px}
.card{background:var(--card);border:1px solid var(--line);border-radius:24px;padding:20px}
.tag{font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#5EEAD4;background:rgba(19,78,74,.5);border:1px solid rgba(20,184,166,.4);padding:3px 9px;border-radius:99px;display:inline-block}
h2{font-family:'Newsreader',serif;font-weight:600;font-size:22px;margin-top:8px}
.muted{color:var(--t3);font-size:12px}
.row{display:flex;align-items:center;justify-content:space-between;gap:8px}
.grp{background:var(--inner);border:1px solid var(--line);border-radius:18px;padding:14px;margin-top:14px}
.lbl{font-size:12px;font-weight:600;color:var(--t2);display:flex;align-items:center;gap:6px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--teal);box-shadow:0 0 6px var(--teal)}
.dot.a{background:var(--amber);box-shadow:0 0 6px var(--amber)}
.unit{font-weight:400;color:var(--t3);font-size:10px}
.ni{background:#0B1220;border:1px solid rgba(20,184,166,.5);color:#99F6E4;border-radius:10px;text-align:center;font:700 14px 'Plus Jakarta Sans',sans-serif;padding:5px 4px;width:72px}
.ni:focus{outline:2px solid var(--teal);outline-offset:0}
.ni.amb{border-color:rgba(245,158,11,.5);color:#FCD34D}
.ni.big{font-size:24px;width:96px}
.sm{width:62px;font-size:12px}
input[type=number]::-webkit-inner-spin-button{opacity:.3}
.btn{background:#1E293B;border:1px solid var(--line2);color:var(--t2);border-radius:10px;font-weight:700;cursor:pointer;transition:.15s}
.btn:hover{border-color:var(--teal);color:#5EEAD4}.btn:active{transform:scale(.96)}
.sq{width:34px;height:34px;font-size:16px}.xs{width:26px;height:26px;font-size:13px}
.chips{display:grid;gap:6px;margin-top:10px}
.c2{grid-template-columns:1fr 1fr}.c3{grid-template-columns:repeat(3,1fr)}.c4{grid-template-columns:repeat(4,1fr)}.c5{grid-template-columns:repeat(5,1fr)}
.chip{background:rgba(30,41,59,.8);border:1px solid var(--line2);color:var(--t2);border-radius:12px;padding:8px 6px;text-align:left;cursor:pointer;font:600 11px 'Plus Jakarta Sans',sans-serif;transition:.15s}
.chip small{display:block;font-weight:400;font-size:9.5px;color:var(--t3);margin-top:2px}
.chip:hover{border-color:var(--teal)}
.chip.on{background:rgba(19,78,74,.55);border-color:var(--teal);color:#99F6E4}
.chip.on.warn{background:rgba(120,53,15,.45);border-color:var(--amber);color:#FDE68A}
.chip.on.bad{background:rgba(136,19,55,.4);border-color:var(--rose);color:#FECDD3}
.seg{display:grid;gap:4px;padding:4px;background:#0B1220;border:1px solid var(--line);border-radius:14px;margin-top:10px}
.seg .chip{text-align:center;border-color:transparent;background:transparent;padding:9px 4px}
.seg .chip.on{background:var(--tealD);color:#fff;border-color:var(--teal)}
.badge{font-size:10px;font-weight:600;padding:3px 9px;border-radius:99px;border:1px solid;white-space:nowrap}
.b-t{color:#5EEAD4;background:rgba(19,78,74,.45);border-color:rgba(20,184,166,.4)}
.b-a{color:#FCD34D;background:rgba(120,53,15,.45);border-color:rgba(245,158,11,.4)}
.b-r{color:#FDA4AF;background:rgba(136,19,55,.4);border-color:rgba(244,63,94,.4)}
.b-s{color:var(--t2);background:#1E293B;border-color:var(--line2)}
.tagoff{font-size:9px;color:var(--t4);border:1px dashed var(--line2);padding:1px 6px;border-radius:99px;display:none}
.tagoff.show{display:inline-block}
.calc{background:var(--deep);border:1px solid var(--line2);border-radius:14px;padding:10px;margin-top:10px}
.in{background:#0B1220;border:1px solid var(--line2);color:var(--t1);border-radius:10px;text-align:center;font:700 12px 'Plus Jakarta Sans',sans-serif;padding:6px 4px;width:100%}
.unitsw{display:inline-flex;background:#1E293B;border:1px solid var(--line2);border-radius:8px;padding:2px}
.unitsw button{border:0;background:transparent;color:var(--t3);font:600 9px 'Plus Jakarta Sans',sans-serif;padding:2px 7px;border-radius:6px;cursor:pointer}
.unitsw button.on{background:#0B1220;color:#5EEAD4}
.ruler{display:flex;justify-content:space-between;align-items:flex-end;height:34px;border:1px solid var(--line2);background:#0B1220;border-radius:10px;padding:4px 8px;margin-top:8px;position:relative}
.ruler i{width:2px;background:var(--line2);display:block}
.ruler b{position:absolute;top:4px;transform:translateX(-50%);background:var(--amber);color:#1c1204;font:700 10px monospace;border-radius:5px;padding:3px 6px;transition:left .2s}
/* results */
#res{position:relative;overflow:hidden;background:linear-gradient(180deg,#0F172A,#111A30 55%,#0A0F1D);transition:opacity .2s}
#res.busy{opacity:.78}
.gauge{position:relative;width:160px;height:160px;margin:12px auto 0}
.gauge svg{transform:rotate(-90deg)}
.gauge .c{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.gauge .n{font-size:38px;font-weight:700;letter-spacing:-.03em}
.up{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--t3);font-weight:500}
.center{text-align:center}
.lvl{display:inline-flex;align-items:center;gap:8px;padding:4px 12px;border-radius:99px;font-size:12px;font-weight:600;border:1px solid;margin-top:12px}
.lvl i{width:8px;height:8px;border-radius:50%}
.lv-t{color:#5EEAD4;background:rgba(20,184,166,.15);border-color:rgba(20,184,166,.35)}.lv-t i{background:var(--teal);box-shadow:0 0 8px var(--teal)}
.lv-a{color:#FCD34D;background:rgba(245,158,11,.15);border-color:rgba(245,158,11,.3)}.lv-a i{background:var(--amber);box-shadow:0 0 8px var(--amber)}
.lv-r{color:#FDA4AF;background:rgba(244,63,94,.15);border-color:rgba(244,63,94,.3)}.lv-r i{background:var(--rose);box-shadow:0 0 8px var(--rose)}
.banner{margin-top:16px;padding:12px;border-radius:16px;background:rgba(19,78,74,.55);border:1px solid rgba(20,184,166,.35);display:flex;justify-content:space-between;gap:10px;align-items:center}
.banner b{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#5EEAD4;display:block}
.banner span{font-size:11px;color:var(--t2)}
.banner .v{font:800 17px monospace;color:#5EEAD4;text-align:right}.banner .v small{display:block;font:500 9px 'Plus Jakarta Sans';color:rgba(94,234,212,.7)}
.hr{border-top:1px solid var(--line);margin-top:18px;padding-top:16px}
.h3{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--t2)}
.drv{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:14px;margin-top:10px}
.drv.hi{border-color:rgba(245,158,11,.3)}
.ico{width:28px;height:28px;border-radius:10px;display:grid;place-items:center;background:rgba(20,184,166,.15);border:1px solid rgba(20,184,166,.3);flex:none;font-size:13px}
.pips{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-top:10px;padding-top:9px;border-top:1px solid rgba(255,255,255,.06)}
.pips i{height:6px;border-radius:9px;background:#1E293B;transition:.3s}
.pips i.t{background:var(--teal);box-shadow:0 0 6px var(--teal)}.pips i.a{background:var(--amber);box-shadow:0 0 6px var(--amber)}.pips i.s{background:#94A3B8}
.what{margin-top:10px;padding:9px 10px 6px;border-radius:12px;background:rgba(15,23,42,.55);border-top:1px solid rgba(255,255,255,.06)}
.what .row{font-size:11px;margin-bottom:4px}
.what .k{color:#2DD4BF;font-weight:600}
.what .scale{display:flex;justify-content:space-between;font:9px monospace;color:var(--t4);margin-top:2px}
.mono{font-family:ui-monospace,Menlo,monospace}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:24px;background:transparent;cursor:pointer}
input[type=range]::-webkit-slider-runnable-track{height:6px;border-radius:9px;background:var(--fill,#1E293B)}
input[type=range]::-moz-range-track{height:6px;border-radius:9px;background:var(--fill,#1E293B)}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:20px;height:20px;margin-top:-7px;border-radius:50%;background:var(--teal);border:2px solid #042F2E;box-shadow:0 0 10px rgba(20,184,166,.6)}
input[type=range]::-moz-range-thumb{width:18px;height:18px;border-radius:50%;background:var(--teal);border:2px solid #042F2E}
.rec{display:flex;gap:12px;padding:12px;border-radius:16px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);margin-top:8px;font-size:12px;line-height:1.5;color:#E2E8F0}
.rec i{width:10px;height:10px;border-radius:50%;background:var(--amber);box-shadow:0 0 6px var(--amber);flex:none;margin-top:5px}
.trust{display:flex;gap:12px;align-items:center;padding:14px;border-radius:18px;background:var(--card);border:1px solid rgba(20,184,166,.3);font-size:11px;line-height:1.5;color:var(--t2)}
.cta{width:100%;border:0;border-radius:16px;padding:14px;font:700 13px 'Plus Jakarta Sans',sans-serif;color:#fff;background:linear-gradient(90deg,#0D9488,#14B8A6);cursor:pointer;box-shadow:0 8px 20px rgba(13,148,136,.25)}
.cta:active{transform:scale(.98)}
.foot{font-size:11px;color:var(--t3);text-align:center;line-height:1.6;padding:0 14px}
.mark{position:absolute;right:-60px;bottom:-60px;width:240px;height:240px;opacity:.07;pointer-events:none;color:var(--teal)}
.hide{display:none!important}
</style></head>
<body><div id="app">
<header>
  <div class="brand"><div class="logo"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2DD4BF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3s6 6.2 6 10.5A6 6 0 0 1 6 13.5C6 9.2 12 3 12 3z"/><path d="M9.5 14l1.8 1.8 3.2-3.6"/></svg></div>
  <div><h1>MadhuRaksha</h1><div class="sub">Precision Diabetes Risk • South Asia</div></div></div>
  <div class="pill"><span class="dots"><i style="background:#FF9933"></i><i style="background:#fff"></i><i style="background:#138808"></i></span>IN</div>
</header>

<section class="card" id="intake">
  <div class="row"><span class="tag">Clinical Diagnostic Intake</span><span class="muted">Live estimate</span></div>
  <h2>Tell us about yourself</h2>
  <p class="muted" style="margin-top:4px">Risk updates live below, using the model trained on your dataset. Tap any number to type exact values.</p>

  <div class="grp" style="padding:10px">
    <div class="row" style="font-size:11px"><span class="lbl">⚡ Quick profile</span><a href="#" id="reset" style="color:#2DD4BF;font-size:10px;font-weight:600;text-decoration:none">Reset defaults</a></div>
    <div class="chips c2" style="margin-top:8px">
      <button class="chip" data-preset="desk">🏢 Urban Desk Worker<small>High sitting, rice/roti, 42y</small></button>
      <button class="chip" data-preset="festive">🪔 Post-Festive / Pre-DM<small>Sweets, waist 36", 50y</small></button>
    </div>
  </div>

  <div class="grp">
    <div class="row"><span class="lbl"><span class="dot"></span>Age <span class="tagoff" data-tag="age">not in model</span></span><span class="badge b-a" id="ageBadge">⚠ Higher risk &gt;35y</span></div>
    <div class="row" style="background:var(--deep);border:1px solid var(--line2);border-radius:14px;padding:10px;margin-top:10px">
      <button class="btn sq" data-step="age" data-d="-1">−</button>
      <div class="row" style="gap:8px"><input class="ni big" type="number" id="age" data-k="age" min="18" max="100" step="1"><span style="font-weight:600;color:var(--t2)">years</span></div>
      <button class="btn sq" data-step="age" data-d="1">+</button>
    </div>
    <div class="chips c5">
      <button class="chip" data-k="age" data-v="28" data-lo="0" data-hi="29" style="text-align:center">20s</button>
      <button class="chip" data-k="age" data-v="35" data-lo="30" data-hi="39" style="text-align:center">30s</button>
      <button class="chip" data-k="age" data-v="45" data-lo="40" data-hi="49" style="text-align:center">40s</button>
      <button class="chip" data-k="age" data-v="55" data-lo="50" data-hi="59" style="text-align:center">50s</button>
      <button class="chip" data-k="age" data-v="65" data-lo="60" data-hi="200" style="text-align:center">60+</button>
    </div>
  </div>

  <div class="grp">
    <div class="row"><span class="lbl"><span class="dot"></span>Body Mass Index <span class="unit">kg/m²</span> <span class="tagoff" data-tag="bmi">not in model</span></span>
      <span class="row" style="gap:4px"><button class="btn xs" data-step="bmi" data-d="-0.5">−</button><input class="ni sm" type="number" id="bmi" data-k="bmi" min="12" max="60" step="0.1"><button class="btn xs" data-step="bmi" data-d="0.5">+</button></span></div>
    <div class="calc">
      <div class="row"><span class="lbl">📐 Calculate from height &amp; weight</span><span class="badge b-t" style="font-size:9px">Auto-syncs BMI</span></div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px">
        <div><div class="row" style="margin-bottom:4px"><span class="muted" style="font-size:10px">Height</span><span class="unitsw"><button id="ucm" class="on">cm</button><button id="uft">ft/in</button></span></div>
          <input class="in" type="number" id="h" min="90" max="240" step="1">
          <div id="ftin" class="hide" style="display:grid;grid-template-columns:1fr 1fr;gap:4px"><input class="in" type="number" id="ft" min="3" max="7" placeholder="ft"><input class="in" type="number" id="inch" min="0" max="11" placeholder="in"></div></div>
        <div><div class="row" style="margin-bottom:4px"><span class="muted" style="font-size:10px">Weight</span><span class="muted" style="font-size:10px">kg</span></div>
          <input class="in" type="number" id="w" min="25" max="250" step="0.5"></div>
      </div>
    </div>
    <div class="chips c2">
      <button class="chip" data-k="bmi" data-v="21.5" data-lo="0" data-hi="22.9">Lean<small>&lt;23 · normal for South Asians</small></button>
      <button class="chip warn" data-k="bmi" data-v="24" data-lo="23" data-hi="24.9">Overweight (Asian)<small>23–24.9</small></button>
      <button class="chip warn" data-k="bmi" data-v="27.5" data-lo="25" data-hi="29.9">Obese I (Asian)<small>25–29.9 · insulin resistance</small></button>
      <button class="chip bad" data-k="bmi" data-v="31" data-lo="30" data-hi="100">Obese II<small>≥30 · high metabolic risk</small></button>
    </div>
  </div>

  <div class="grp">
    <div class="row"><span class="lbl"><span class="dot a"></span>Abdominal Adiposity (Waist) <span class="tagoff" data-tag="waist">not in model</span></span>
      <span class="row" style="gap:4px"><input class="ni amb sm" type="number" id="waist" data-k="waist" min="20" max="65" step="1"><span class="mono" style="font-size:11px;color:#FCD34D" id="waistcm"></span></span></div>
    <div class="row muted mono" style="font-size:10px;margin-top:8px"><span>28" slim</span><span style="color:#FB7185">⚠ cutoff ≈ 31" F / 35" M</span><span>42" visceral</span></div>
    <div class="ruler" id="ruler"><i style="height:16px"></i><i style="height:8px"></i><i style="height:8px"></i><i style="height:12px;background:var(--amber)"></i><i style="height:8px"></i><i style="height:8px"></i><i style="height:8px"></i><i style="height:16px;background:#FB7185"></i><i style="height:8px"></i><i style="height:16px"></i><b id="rulerb">34"</b></div>
    <div class="chips c5">
      <button class="chip warn" data-k="waist" data-v="28" data-lo="0" data-hi="29" style="text-align:center">28"</button>
      <button class="chip warn" data-k="waist" data-v="31" data-lo="30" data-hi="32" style="text-align:center">31"</button>
      <button class="chip warn" data-k="waist" data-v="34" data-lo="33" data-hi="35" style="text-align:center">34"</button>
      <button class="chip warn" data-k="waist" data-v="36" data-lo="36" data-hi="38" style="text-align:center">36"</button>
      <button class="chip warn" data-k="waist" data-v="40" data-lo="39" data-hi="100" style="text-align:center">40"</button>
    </div>
  </div>

  <div class="grp">
    <div class="row"><span class="lbl"><span class="dot a"></span>Fasting Blood Glucose <span class="unit">mg/dL</span> <span class="tagoff" data-tag="glucose">not in model</span></span><span class="badge b-a" id="glBadge"></span></div>
    <div class="row" style="background:#020617;border:1px solid var(--line);border-radius:14px;padding:10px;margin-top:10px">
      <span class="row" style="gap:8px"><span style="font-size:18px">🩸</span><span><input class="ni big mono" type="number" id="glucose" data-k="glucose" min="50" max="350" step="1" style="width:100px;font-size:22px"> <span class="mono" style="color:#2DD4BF;font-weight:700">mg/dL</span></span></span>
      <span class="row" style="gap:6px"><button class="btn sq" data-step="glucose" data-d="-2">−</button><button class="btn sq" data-step="glucose" data-d="2">+</button></span>
    </div>
    <div class="chips c4">
      <button class="chip" data-k="glucose" data-v="92" data-lo="0" data-hi="99" style="text-align:center">92<small>Normal</small></button>
      <button class="chip warn" data-k="glucose" data-v="110" data-lo="100" data-hi="115" style="text-align:center">110<small>Pre-DM</small></button>
      <button class="chip warn" data-k="glucose" data-v="118" data-lo="116" data-hi="125" style="text-align:center">118<small>Mild</small></button>
      <button class="chip bad" data-k="glucose" data-v="132" data-lo="126" data-hi="1000" style="text-align:center">132<small>Diabetic range</small></button>
    </div>
  </div>

  <div class="grp">
    <div class="row"><span class="lbl"><span class="dot" style="background:#FB7185;box-shadow:0 0 6px #FB7185"></span>Blood Pressure (Systolic) <span class="tagoff" data-tag="bp">not in model</span></span>
      <span class="row" style="gap:4px"><button class="btn xs" data-step="bp" data-d="-2">−</button><input class="ni sm" type="number" id="bp" data-k="bp" min="70" max="240" step="1"><span class="muted" style="font-size:10px">mmHg</span><button class="btn xs" data-step="bp" data-d="2">+</button></span></div>
    <div class="chips c3">
      <button class="chip" data-k="bp" data-v="118" data-lo="0" data-hi="119">Optimal<small>&lt;120</small></button>
      <button class="chip warn" data-k="bp" data-v="134" data-lo="120" data-hi="139">Pre-HTN<small>120–139</small></button>
      <button class="chip bad" data-k="bp" data-v="144" data-lo="140" data-hi="400">Stage 1 HTN<small>≥140</small></button>
    </div>
  </div>

  <div class="grp">
    <div class="row"><span class="lbl">🍚 Daily diet &amp; carbs <span class="tagoff show" data-tag="diet">not in model</span></span></div>
    <div class="chips c2">
      <button class="chip" data-k="diet" data-v="rice_heavy">🍚 Polished rice heavy<small>High glycemic spikes</small></button>
      <button class="chip" data-k="diet" data-v="balanced">🫓 Balanced roti / dal<small>Moderate GI load</small></button>
      <button class="chip" data-k="diet" data-v="millets">🌾 Millet &amp; high fibre<small>Slow-release carbs</small></button>
      <button class="chip" data-k="diet" data-v="sweets">☕ Chai &amp; mithai often<small>≥3 sweet cups/week</small></button>
    </div>
  </div>

  <div class="grp">
    <div class="row"><span class="lbl">👪 Family diabetic lineage <span class="tagoff" data-tag="family">not in model</span></span></div>
    <div class="seg" style="grid-template-columns:repeat(4,1fr)">
      <button class="chip" data-k="family" data-v="none">None</button><button class="chip" data-k="family" data-v="1parent">1 Parent</button><button class="chip" data-k="family" data-v="both">Both</button><button class="chip" data-k="family" data-v="sibling">Sibling</button>
    </div>
  </div>

  <div class="grp">
    <div class="row"><span class="lbl">⚡ Physical activity &amp; desk strain <span class="tagoff" data-tag="activity">not in model</span></span><span class="badge b-t" id="actBadge"></span></div>
    <div class="seg" style="grid-template-columns:repeat(3,1fr)">
      <button class="chip" data-k="activity" data-v="low">Low (&lt;30m)</button><button class="chip" data-k="activity" data-v="medium">Med (30–45m)</button><button class="chip" data-k="activity" data-v="high">Active (45m+)</button>
    </div>
    <p class="muted" style="font-size:10px;margin-top:8px">Tip: break up long sitting every 45 minutes.</p>
  </div>
</section>

<section class="card" id="res">
  <svg class="mark" viewBox="0 0 100 100" fill="none" stroke="currentColor"><circle cx="50" cy="50" r="45" stroke-dasharray="2 2" stroke-width=".8"/><circle cx="50" cy="50" r="35" stroke-width=".6"/><circle cx="50" cy="50" r="25" stroke-width=".5"/><path d="M50 5Q65 50 50 95Q35 50 50 5" stroke-width=".6"/><path d="M5 50Q50 65 95 50Q50 35 5 50" stroke-width=".6"/></svg>
  <div class="center">
    <span class="tag">Dynamic Clinical Assessment</span>
    <div class="gauge">
      <svg viewBox="0 0 120 120" width="160" height="160"><circle cx="60" cy="60" r="48" fill="none" stroke="#1E293B" stroke-width="10"/>
        <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop id="g0" offset="0" stop-color="#F59E0B"/><stop id="g1" offset="1" stop-color="#EA580C"/></linearGradient></defs>
        <circle id="arc" cx="60" cy="60" r="48" fill="none" stroke="url(#g)" stroke-width="11" stroke-linecap="round" stroke-dasharray="301.6" stroke-dashoffset="301.6" style="transition:stroke-dashoffset .5s ease-out"/></svg>
      <div class="c"><span class="n" id="pct">–</span><span class="up">Estimated risk</span></div>
    </div>
    <div class="lvl lv-a" id="lvl"><i></i><span id="lvlt">Calculating…</span></div>
    <h2 id="lvlh" style="margin-top:6px">&nbsp;</h2>
    <p class="muted" style="max-width:290px;margin:6px auto 0;line-height:1.5;color:var(--t2)" id="inHundred"></p>
  </div>

  <div class="banner"><div><b>Projected risk reduction</b><span id="bnText">Adjust the what-if targets below.</span></div><div class="v"><span id="bnVal" style="color:inherit;font:inherit">0</span><small>points</small></div></div>

  <div class="hr">
    <div class="row" style="align-items:flex-start"><div><span class="h3">Biomarker impact spectrum</span><p class="muted" style="font-size:10px;margin-top:3px">Bars show how much each input adds versus a healthy reference value. Drag the what-if sliders to test changes.</p></div>
    <button class="btn" id="resetSc" style="font-size:9px;padding:5px 9px;flex:none">↺ Reset</button></div>
    <div id="cards"></div>
  </div>

  <div class="hr"><span class="h3" style="color:var(--t3)">Priority recommendations</span><div id="recs"></div></div>
</section>

<div class="trust"><div class="ico" style="width:32px;height:32px;border-radius:50%">🛡</div><div>Uses Asian BMI cut-offs (23 / 25) and an earlier-onset age flag (&gt;35). Risk values come from the model trained on the dataset behind this app and are an estimate, not a diagnosis.</div></div>
<button class="cta" id="dl">⬇ Download summary for your doctor</button>
<p class="foot">A screening aid, not a clinical diagnosis. Your inputs are not saved or stored. Please consult a registered medical practitioner for evaluation.</p>
</div>

<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const D={age:45,h:170,w:79.5,bmi:27.5,waist:34,glucose:118,bp:134,family:'1parent',activity:'low',diet:'rice_heavy'};
const PRE={desk:{age:42,bmi:26.5,waist:33,glucose:112,bp:130,family:'1parent',activity:'low',diet:'rice_heavy'},
           festive:{age:50,bmi:28.5,waist:36,glucose:124,bp:138,family:'both',activity:'low',diet:'sweets'}};
let st={...D}, sc={}, res=null, seq=0, timer=null, streamlit=null;
const scen=()=>({...st,...sc});
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));

/* ---------- Streamlit protocol ---------- */
const send=(type,o)=>window.parent.postMessage({isStreamlitMessage:true,type,...o},"*");
const frame=()=>send("streamlit:setFrameHeight",{height:document.documentElement.scrollHeight+8});
window.addEventListener("message",e=>{
  if(e.data.type!=="streamlit:render")return;
  const r=e.data.args.result; if(r&&r.seq===seq){res=r;renderRes();}
  frame();
});
new ResizeObserver(frame).observe(document.body);
send("streamlit:componentReady",{apiVersion:1});

function push(){
  clearTimeout(timer);
  timer=setTimeout(()=>{
    seq++; $('#res').classList.add('busy');
    const pick=o=>({age:o.age,bmi:o.bmi,waist:o.waist,glucose:o.glucose,bp:o.bp,family:o.family,activity:o.activity});
    send("streamlit:setComponentValue",{value:{seq,st:pick(st),sc:pick(scen())},dataType:"json"});
  },140);
}

/* ---------- inputs ---------- */
function setK(k,v){
  if(typeof D[k]==='number'){v=parseFloat(v); if(isNaN(v))return;}
  st[k]=v;
  if(k==='bmi'){/* manual BMI keeps h/w as typed */}
  sc={}; syncIn(); push();
}
function syncIn(){
  $$('input[data-k]').forEach(i=>{ if(document.activeElement!==i) i.value=st[i.dataset.k]; });
  if(document.activeElement!==$('#h'))$('#h').value=st.h;
  if(document.activeElement!==$('#w'))$('#w').value=st.w;
  $$('.chip[data-k]').forEach(c=>{
    const k=c.dataset.k, v=st[k]; let on;
    if(c.dataset.lo!==undefined) on=v>=+c.dataset.lo&&v<=+c.dataset.hi; else on=String(v)===c.dataset.v;
    c.classList.toggle('on',on);
  });
  $('#ageBadge').textContent=st.age>35?`⚠ +${st.age-35}y beyond 35`:'Below 35y';
  const g=st.glucose;
  $('#glBadge').textContent=g<100?'Normal':g<=125?'Prediabetes band':'Diabetes range';
  $('#glBadge').className='badge '+(g<100?'b-t':g<=125?'b-a':'b-r');
  $('#actBadge').textContent={low:'Low active (<30m)',medium:'Moderate (30–45m)',high:'Active (45m+)'}[st.activity];
  $('#waistcm').textContent=`in (${Math.round(st.waist*2.54)} cm)`;
  $('#rulerb').textContent=st.waist+'"'; $('#rulerb').style.left=clamp((st.waist-26)/(44-26),0,1)*88+6+'%';
  syncSc();
}
$$('input[data-k]').forEach(i=>i.addEventListener('input',()=>setK(i.dataset.k,i.value)));
$$('.chip[data-k]').forEach(c=>c.addEventListener('click',()=>setK(c.dataset.k,c.dataset.v)));
$$('[data-step]').forEach(b=>b.addEventListener('click',()=>{const k=b.dataset.step;setK(k,+(st[k]+ +b.dataset.d).toFixed(1));}));
$$('[data-preset]').forEach(b=>b.addEventListener('click',()=>{Object.assign(st,PRE[b.dataset.preset]);st.h=170;st.w=+(st.bmi*1.7*1.7).toFixed(1);sc={};syncIn();push();}));
$('#reset').addEventListener('click',e=>{e.preventDefault();st={...D};sc={};syncIn();push();});
function hw(){const h=parseFloat($('#h').value),w=parseFloat($('#w').value);if(!(h>0&&w>0))return;st.h=h;st.w=w;st.bmi=+(w/((h/100)**2)).toFixed(1);sc={};syncIn();push();}
$('#h').addEventListener('input',hw);$('#w').addEventListener('input',hw);
function ftin(){const cm=Math.round(((+$('#ft').value||0)*12+(+$('#inch').value||0))*2.54);if(cm>0){$('#h').value=cm;hw();}}
$('#ft').addEventListener('input',ftin);$('#inch').addEventListener('input',ftin);
function unit(u){$('#ucm').classList.toggle('on',u==='cm');$('#uft').classList.toggle('on',u==='ft');
  $('#h').classList.toggle('hide',u==='ft');$('#ftin').classList.toggle('hide',u==='cm');
  if(u==='ft'){const t=st.h/2.54;$('#ft').value=Math.floor(t/12);$('#inch').value=Math.round(t%12);}}
$('#ucm').onclick=()=>unit('cm');$('#uft').onclick=()=>unit('ft');

/* ---------- driver cards ---------- */
const CARDS=[
 {k:'glucose',t:'Fasting Glucose',ico:'🧪',sim:{min:70,max:200,step:1,unit:'mg/dL',lo:'70 (optimal)',mid:'95 (ideal)',hi:'200'}},
 {k:'family',t:'Family History',ico:'👪',fixed:true},
 {k:'bmi',t:'BMI & Central Waist',ico:'⚖',sim:{min:16,max:40,step:.5,unit:'kg/m²',lo:'16',mid:'22.8 (optimal SA BMI)',hi:'40'}},
 {k:'activity',t:'Movement',ico:'⚡',act:true},
 {k:'bp',t:'Systolic BP',ico:'🫀',sim:{min:90,max:180,step:1,unit:'mmHg',lo:'90',mid:'118 (optimal)',hi:'180'}}];
$('#cards').innerHTML=CARDS.map(c=>`
<div class="drv" id="c-${c.k}"><div class="row"><div class="row" style="gap:10px;justify-content:flex-start"><div class="ico">${c.ico}</div>
 <div><div style="font-size:12px;font-weight:600">${c.t} <span class="mono" style="font-size:10px;color:#FCD34D;font-weight:500" id="v-${c.k}"></span></div><div class="muted" style="font-size:10px;margin-top:2px" id="d-${c.k}"></div></div></div>
 <div style="text-align:right"><span class="badge" id="b-${c.k}"></span><div class="mono" style="font-size:10px;font-weight:700;margin-top:4px" id="x-${c.k}"></div></div></div>
 <div class="pips" id="p-${c.k}"></div>
 ${c.sim?`<div class="what"><div class="row"><span><span class="k">What-if:</span> target</span><span><s class="muted" id="sp-${c.k}"></s> <b class="mono" style="color:#5EEAD4" id="sv-${c.k}"></b></span></div>
 <input type="range" id="s-${c.k}" min="${c.sim.min}" max="${c.sim.max}" step="${c.sim.step}"><div class="scale"><span>${c.sim.lo}</span><span style="color:#2DD4BF">${c.sim.mid}</span><span>${c.sim.hi}</span></div></div>`:''}
 ${c.act?`<div class="what"><div class="row"><span><span class="k">What-if:</span> activity target</span></div><div class="seg" style="grid-template-columns:repeat(3,1fr);margin-top:6px">
  ${['low','medium','high'].map(a=>`<button class="chip" data-sa="${a}">${{low:'Low',medium:'Med',high:'Active'}[a]}</button>`).join('')}</div></div>`:''}
 ${c.fixed?`<div class="what row" style="font-size:10px;color:var(--t3)"><span>● Fixed / non-modifiable baseline</span><i>Lifestyle can offset genetic load</i></div>`:''}
</div>`).join('');
CARDS.filter(c=>c.sim).forEach(c=>$('#s-'+c.k).addEventListener('input',e=>{sc[c.k]=+e.target.value;syncSc();push();}));
$$('[data-sa]').forEach(b=>b.addEventListener('click',()=>{sc.activity=b.dataset.sa;syncSc();push();}));
$('#resetSc').onclick=()=>{sc={};syncSc();push();};
function syncSc(){
  const s=scen();
  CARDS.filter(c=>c.sim).forEach(c=>{
    const el=$('#s-'+c.k); if(!el)return; el.value=s[c.k];
    const p=(s[c.k]-c.sim.min)/(c.sim.max-c.sim.min)*100; el.style.setProperty('--fill',`linear-gradient(to right,#14B8A6 ${p}%,#1E293B ${p}%)`);
    $('#sv-'+c.k).textContent=`${c.k==='bmi'?(+s[c.k]).toFixed(1):s[c.k]} ${c.sim.unit}`;
    $('#sp-'+c.k).textContent=sc[c.k]!==undefined&&sc[c.k]!==st[c.k]?st[c.k]:'';
  });
  $$('[data-sa]').forEach(b=>b.classList.toggle('on',b.dataset.sa===s.activity));
}

/* ---------- results ---------- */
const pips=(n,cls)=>[1,2,3,4,5].map(i=>`<i class="${i<=n?cls:''}"></i>`).join('');
function impact(d){return d>=6?['High impact','b-a','a',4]:d>=2?['Moderate','b-t','t',3]:d>-1?['Low','b-s','s',1]:['Protective','b-t','t',1];}
function renderRes(){
  if(!res)return; const r=res, d=r.drivers;
  $('#res').classList.remove('busy');
  $('#pct').textContent=r.risk.toFixed(0)+'%';
  $('#arc').style.strokeDashoffset=301.6*(1-Math.min(r.risk,100)/100);
  const L={Low:['lv-t','Low risk','#14B8A6','#059669'],Medium:['lv-a','Moderate risk','#F59E0B','#EA580C'],High:['lv-r','High risk','#EF4444','#F97316']}[r.level];
  $('#lvl').className='lvl '+L[0]; $('#lvlt').textContent=L[1]; $('#lvlh').textContent=L[1];
  $('#g0').setAttribute('stop-color',L[2]);$('#g1').setAttribute('stop-color',L[3]);
  $('#inHundred').innerHTML=`The model estimates about <b style="color:#fff">${r.risk.toFixed(0)} in 100</b> people with this profile have diabetes (training-data average: ${r.prev.toFixed(0)} in 100).`;
  const dd=r.risk-r.scRisk;
  $('#bnVal').textContent=(dd>0.05?'−':dd<-0.05?'+':'')+Math.abs(dd).toFixed(1);
  $('#bnVal').parentElement.style.color=dd<-0.05?'#FCD34D':'#5EEAD4';
  $('#bnText').textContent=Math.abs(dd)<0.05?'Adjust the what-if targets below.':dd>0?`Targets lower risk from ${r.risk.toFixed(1)}% to ${r.scRisk.toFixed(1)}%.`:`This scenario raises risk to ${r.scRisk.toFixed(1)}%.`;
  const s=st;
  const gd=s.glucose<100?'Normal fasting glucose (<100 mg/dL)':s.glucose<=125?'Prediabetes range (100–125 mg/dL)':'Diabetes range (≥126 mg/dL)';
  const bd=(s.bmi<23?'Normal for South Asians':s.bmi<25?'Overweight (Asian cut-off ≥23)':'Obese (Asian cut-off ≥25)')+` • waist ${s.waist}"`;
  const pd=s.bp<120?'Optimal (<120 mmHg)':s.bp<140?'Pre-hypertension range (120–139)':'Stage 1 hypertension or higher (≥140)';
  const fam={none:'None',"1parent":'1 parent',both:'Both parents',sibling:'Sibling'}[s.family];
  const info={glucose:[s.glucose+' mg/dL',gd],family:[fam,s.family==='none'?'No first-degree history reported':'First-degree history reported'],
    bmi:[s.bmi.toFixed(1)+' kg/m²',bd],activity:[{low:'Low',medium:'Moderate',high:'Active'}[s.activity],{low:'<30 min active movement per day',medium:'Meets basic recommendation',high:'Protective level of activity'}[s.activity]],bp:[s.bp+' mmHg',pd]};
  CARDS.forEach(c=>{
    const dv=d[c.k]||0, [lab,bc,pc,n]=impact(dv);
    $('#v-'+c.k).textContent=info[c.k][0]; $('#d-'+c.k).textContent=info[c.k][1];
    $('#b-'+c.k).textContent=lab; $('#b-'+c.k).className='badge '+bc;
    $('#x-'+c.k).textContent=(dv>0?'+':dv<0?'−':'')+Math.abs(dv).toFixed(1)+' pts';
    $('#x-'+c.k).style.color=dv>=2?'#FBBF24':'#CBD5E1';
    $('#p-'+c.k).innerHTML=pips(n,pc);
    $('#c-'+c.k).classList.toggle('hi',dv>=6);
  });
  $('#recs').innerHTML=(r.advice||[]).map(t=>`<div class="rec"><i></i><div>${t}</div></div>`).join('');
  const un=new Set([...(r.unused||[]),'diet']);
  $$('.tagoff').forEach(t=>t.classList.toggle('show',un.has(t.dataset.tag)));
  frame();
}
$('#dl').onclick=()=>{
  const s=st,r=res; const lines=['MadhuRaksha – diabetes risk screening summary','',
   r?`Estimated risk: ${r.risk.toFixed(1)}% (${r.level})`:'Risk not calculated yet','',
   `Age: ${s.age}`,`BMI: ${s.bmi} kg/m2`,`Waist: ${s.waist} in`,`Fasting glucose: ${s.glucose} mg/dL`,`Systolic BP: ${s.bp} mmHg`,
   `Family history: ${s.family}`,`Activity: ${s.activity}`,`Diet: ${s.diet}`,'',
   'Screening estimate from a statistical model. Not a diagnosis. Please confirm with a blood test and a doctor.'];
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/plain'}));a.download='madhuraksha-summary.txt';a.click();
};
syncIn(); push();
</script></body></html>
"""


@st.cache_resource
def _ui_component():
    """Write the embedded UI to a temp folder once, so deployment needs only this one file."""
    d = Path(tempfile.gettempdir()) / "madhuraksha_ui"
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.html").write_text(UI_HTML, encoding="utf-8")
    return components.declare_component("madhuraksha_ui", path=str(d))


ui = _ui_component()

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
