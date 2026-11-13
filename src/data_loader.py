from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
LOCAL_CSV = DATA_DIR / "ai4i2020.csv"

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"


def load_dataset() -> pd.DataFrame:
    """Load a local AI4I CSV, or fetch it from the UCI repository."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if LOCAL_CSV.exists():
        df = pd.read_csv(LOCAL_CSV)
    else:
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=601)
            df = pd.concat(
                [dataset.data.features, dataset.data.targets],
                axis=1
            )
            # Keep a local copy for repeatable subsequent runs.
            df.to_csv(LOCAL_CSV, index=False)
        except Exception as exc:
            raise RuntimeError(
                "Could not download AI4I from UCI. Check your internet "
                "connection or download ai4i2020.csv from the official UCI "
                "page and place it at data/ai4i2020.csv."
            ) from exc

    # Some distributions name the first identifier column differently;
    # only the explicitly listed predictors and target are required.
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            "Use the official AI4I 2020 CSV."
        )
    return df[required].dropna().copy()
