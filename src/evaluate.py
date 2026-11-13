"""Evaluate the saved model on a fresh stratified hold-out split.

For a formal experiment, prefer evaluating during `python -m src.train`,
which writes the comparison table from the exact same test split.
"""
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
from src.data_loader import load_dataset, FEATURES, TARGET


def main():
    model_path = Path("models/best_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError("Train first: python -m src.train")

    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET].astype(int),
        test_size=0.20, stratify=df[TARGET].astype(int), random_state=42
    )
    model = joblib.load(model_path)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)
    print("Classification report (threshold = 0.50):")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print(f"PR-AUC: {average_precision_score(y_test, probabilities):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")


if __name__ == "__main__":
    main()
