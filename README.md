# MadhuRaksha: AI-Powered Diabetes Risk Screening for India

MadhuRaksha is an AI-powered diabetes risk screening app for India. Enter your age, BMI, glucose and blood pressure to get a Low, Medium or High risk level, the factors raising your risk, and prevention advice. For education only, not a medical diagnosis. No inputs are stored.

**Live app:** `PASTE-YOUR-STREAMLIT-LINK-HERE`

> The name comes from *madhumeha* (diabetes) and *raksha* (protection).

## Why this project

India has one of the largest diabetes burdens in the world, and many people are never tested until complications appear. This college R&D project explores how a simple data-driven tool can help people notice their risk early and get tested in time.

## Features

- **Risk level and probability:** Low, Medium or High, with an estimated percentage.
- **Top risk factors:** shows which inputs raise your risk most compared with a typical person.
- **Prevention advice:** tips that change with the risk level.
- **"What if" simulator:** see how your risk changes if you change one input, such as BMI.
- **Asian BMI categories:** overweight from 23 and obese from 25.
- **Model transparency:** the "About the model" tab shows test scores and limitations.
- **Privacy:** user inputs are not saved or stored.

## How it works

1. The data is cleaned (impossible zeros in fields like glucose and BMI are treated as missing values).
2. A Gradient Boosting model is trained with probability calibration, so a predicted 30% risk reflects an actual likelihood of about 30%.
3. The model is tested on a held-out 20% of the data and scored with ROC-AUC and PR-AUC.
4. The Streamlit app turns the probability into a risk level, explains the result and gives advice.

Risk-level cut-offs are adjustable in the app sidebar and should be justified in the report.

## Dataset

The app currently uses the public **Pima Indians Diabetes** dataset (768 rows). Despite the name, this data comes from Pima Native American women in Arizona, USA, not from India. It is used here for development and demonstration.

The project is designed to be retrained on Indian survey data such as NFHS-5 or LASI once access is approved. Those files cannot be shared publicly, so they are not included in this repository.

## Run it yourself

```bash
pip install -r requirements.txt
streamlit run app.py
```

Put your cleaned CSV next to `app.py` and name it `data.csv`. The app picks it up automatically. It looks for a diabetes column such as `Outcome`, and you can change this under "Data settings (advanced)" in the sidebar. Use numeric columns only.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app, including model training |
| `data.csv` | Cleaned dataset used by the app |
| `requirements.txt` | Python packages needed |

## Limitations

- It estimates current risk from the inputs given. It does not predict future years.
- Accuracy depends on the dataset, and a small dataset gives less stable results.
- The training data is not from India, so results may not reflect Indian patients.
- Self-reported or survey-style inputs can be inaccurate.

## Future work

- Retrain and validate on Indian data (NFHS-5, LASI)
- Compare against the Indian Diabetes Risk Score
- Screening mode that needs no blood test
- SHAP explanations and risk ranges
- Hindi and regional language support
- Downloadable PDF summary

## Disclaimer

MadhuRaksha is for education and awareness only. It estimates risk and does not diagnose diabetes. Please confirm with a blood test and consult a doctor.

## Author

`YOUR NAME` | `YOUR COLLEGE` | `YEAR`

## Data source and credits

Pima Indians Diabetes Database (UCI Machine Learning Repository / Kaggle). Built with Python, scikit-learn and Streamlit.
