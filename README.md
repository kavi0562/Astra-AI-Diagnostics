# Astra AI Diagnostics —Platform

This repository contains a simulated Clinical Decision Support (CDS) demo called "Astra AI Diagnostics". It includes a static web-based UI (index.html) that demonstrates a multimodal AI diagnostics interface, and a standalone Python script (predictive_model.py) that generates synthetic data and trains simple machine-learning models for offline experimentation.

> WARNING: This project is a simulation and for demonstration/educational purposes only. It is NOT intended for clinical use.

Contents
- index.html — Static frontend demo (Tailwind CSS + vanilla JS). Run in a browser or serve with a static HTTP server.
- predictive_model.py — CLI tool that generates synthetic patient data, trains RandomForest models, and provides an interactive text-based assessment loop.

Features
- Simulated multimodal inputs (structured vitals, EKG/scan upload placeholders).
- Heuristic model training in the browser (frontend) and more structured ML training in Python (predictive_model.py).
- Local browser history for demo assessments (frontend) and a CLI patient history for the Python script.
- Explainable AI (XAI) style outputs and recommended next steps / precautions (simulated).

Quick start
1. Create and activate a Python virtual environment (recommended):

   python3 -m venv .venv
   source .venv/bin/activate  # macOS / Linux

2. Install Python dependencies:

   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

3. Run the interactive Python demo (CLI):

   python predictive_model.py

   The script will generate synthetic data, train three RandomForest models (diabetes, heart, cancer), and present a simple text menu for running assessments and viewing a patient history.

4. Open the frontend demo in your browser:

   - Option A (quick): Double-click index.html or open it via your browser's File > Open.
   - Option B (recommended): Serve the folder so browser requests behave like a web app:

     python3 -m http.server 8000

     Then open http://localhost:8000/index.html

Notes
- The frontend uses client-side heuristics and synthetic-model training (pure JS). The Python script trains scikit-learn models on generated synthetic data.
- All patient data in this demo is synthetic and stored locally (browser localStorage for the UI; Python script keeps a runtime patient_history list).
- If you intend to integrate a real backend, validate clinical safety, and implement proper privacy, encryption, and regulatory compliance before using any models in production.

Development
- Edit index.html to tweak UI, add features, or connect to a real backend API.
- predictive_model.py can be adapted into a Flask/FastAPI service if you want an HTTP API endpoint for model inference.

License & Disclaimer
- This repository is provided for educational/demo purposes only. Not for clinical decision making.
- No warranty is provided; use at your own risk.

