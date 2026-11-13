from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, accuracy_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data_loader import load_dataset, FEATURES, TARGET

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
RANDOM_STATE = 42


def build_preprocessor():
    numeric = FEATURES[1:]
    categorical = ["Type"]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / max(positives, 1)

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=450, max_depth=5, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=2.0, scale_pos_weight=scale_pos_weight,
            objective="binary:logistic", eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    rows = []
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ])
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.50).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "pr_auc": average_precision_score(y_test, probabilities),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "threshold": 0.50,
        }
        rows.append(row)
        fitted[name] = pipeline
        print(
            f"{name}: PR-AUC={row['pr_auc']:.4f}, "
            f"ROC-AUC={row['roc_auc']:.4f}, F1={row['f1']:.4f}"
        )

    results = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    results.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    best_name = str(results.iloc[0]["model"])
    best_pipeline = fitted[best_name]
    joblib.dump(best_pipeline, MODEL_DIR / "best_model.joblib")

    metadata = {
        "best_model": best_name,
        "features": FEATURES,
        "target": TARGET,
        "threshold": 0.50,
        "random_state": RANDOM_STATE,
        "dataset": "UCI AI4I 2020 Predictive Maintenance Dataset",
        "selection_metric": "PR-AUC (Average Precision)",
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nModel comparison:")
    print(results.to_string(index=False))
    print(f"\nSaved best model ({best_name}) to models/best_model.joblib")
    print("Saved metrics to reports/model_comparison.csv")


if __name__ == "__main__":
    main()
