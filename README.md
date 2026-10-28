# NeuroCheck — Parkinson's Disease Detection (Educational Demo)

A Streamlit website project for VS Code demonstrating a machine-learning classification workflow on voice-measurement features.

## Important safety note
This is an educational prototype, not a medical device. It must not be used to diagnose, rule out, screen for, or guide treatment of Parkinson's disease. The included CSV is **synthetic demonstration data**, not clinical data; model metrics have no clinical meaning.

## Run locally (Windows / VS Code)
1. Install Python 3.10+.
2. Extract the ZIP and open this folder in VS Code.
3. In PowerShell, run:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```
   If activation is blocked, you can skip activation and run `python -m pip install -r requirements.txt` followed by `python -m streamlit run app.py`.
4. Open the local URL shown, usually `http://localhost:8501`.

## Dataset format
CSV needs numeric features and a target column named `status` with values 0 and 1. The included synthetic dataset is provided only so the application runs out of the box. Replace it with a documented, ethically sourced dataset for meaningful academic research and disclose its limitations.

## Files
- `app.py` — Streamlit website and ML pipeline
- `requirements.txt` — dependencies
- `data/parkinsons_sample.csv` — synthetic demo data

## Deploy
Can be deployed as a Streamlit app using a compatible Python hosting service. For Render, use build command `pip install -r requirements.txt` and start command:
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
