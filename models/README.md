# Model artifacts

Run `python -m src.train` to generate `best_model.joblib` and `model_metadata.json`.

Generated model files are excluded from Git to keep the repository lightweight and avoid committing artifacts that can be reproduced from source. The Streamlit app displays instructions if the model has not yet been trained.
