from pathlib import Path
import joblib
import pandas as pd

CANCER = Path(__file__).resolve().parent

MODEL = CANCER / "best_model.pkl"
COLS = CANCER / "feature_columns.pkl"
SCALER = CANCER / "scaler.pkl"

_model = None
_cols = None
_scaler = None


def load_artifacts():
    global _model, _cols, _scaler

    if _model is None:
        _model = joblib.load(MODEL)

    if _cols is None:
        _cols = joblib.load(COLS)

    if _scaler is None:
        _scaler = joblib.load(SCALER)

    return _model, _cols, _scaler


def preprocess(payload: dict):
    load_artifacts()

    payload["concave points_mean"] = payload.pop("concave_points_mean")
    payload["concave points_se"] = payload.pop("concave_points_se")
    payload["concave points_worst"] = payload.pop("concave_points_worst")

    df = pd.DataFrame([payload])

    df = df[_cols]

    df[_cols] = _scaler.transform(df[_cols])

    return df


def predict(payload: dict):
    model, _, _ = load_artifacts()

    df = preprocess(payload)

    prediction = model.predict(df)

    prediction = prediction[0]

    return "Cancer" if int(prediction) == 1 else "No Cancer"