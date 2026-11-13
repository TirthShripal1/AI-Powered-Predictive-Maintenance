from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
from src.data_loader import load_dataset, FEATURES

MODEL_PATH = Path("models/best_model.joblib")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train first: python -m src.train")

    df = load_dataset()
    X = df[FEATURES].sample(min(500, len(df)), random_state=42)
    pipeline = joblib.load(MODEL_PATH)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(X)

    # TreeExplainer supports tree-based models. Logistic Regression is
    # intentionally skipped rather than presenting an invalid explanation.
    if classifier.__class__.__name__ not in {"RandomForestClassifier", "XGBClassifier"}:
        raise ValueError(
            "The selected best model is not tree-based. Select a tree model "
            "or adapt this script to use an appropriate SHAP explainer."
        )

    explainer = shap.TreeExplainer(classifier)
    values = explainer(transformed)
    feature_names = preprocessor.get_feature_names_out()
    shap.plots.beeswarm(values, max_display=15, show=False)
    Path("reports").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", dpi=160, bbox_inches="tight")
    print("Saved reports/shap_summary.png")


if __name__ == "__main__":
    main()
