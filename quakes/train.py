"""Part 2, Task D: train a model on the ETL output and save it.

Command line:
    python -m quakes.train --task regression --model linear
    python -m quakes.train --task regression --model poly --degree 3 --alpha 10
    python -m quakes.train --task classification --model logistic [--balanced]

Steps (see the lab handout):
 1. load the table from the SQLite database written by quakes.etl
 2. X = everything except 'mag' and 'big_quake'; y = 'mag' or 'big_quake'
 3. train/test split (stratified for classification, seeded with config.SEED)
 4. fit the dummy baseline AND your model; report both on the test set
 5. cross-validate on the TRAIN set only
 6. classification only: pick the threshold from out-of-fold training
    predictions (cross_val_predict, method='predict_proba'), then apply it to test
 7. save the pipeline to models/ and the report (JSON) to reports/
"""
import argparse
import json
import sqlite3

import joblib
import pandas as pd
from sklearn.model_selection import cross_val_predict, train_test_split

from . import config
from .etl import DB_PATH
from .evaluate import (best_threshold, classification_report_dict,
                       cross_validate_model, regression_report)
from .models import make_classifier, make_regressor

MODELS_DIR = config.ROOT / "models"
REPORTS_DIR = config.ROOT / "reports"
REGRESSION_MODELS = ["linear", "ridge", "poly"]
CLASSIFICATION_MODELS = ["logistic"]


def load_data(db_path=None):
    db_path = db_path or DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} not found. Run: python -m quakes.etl --mode batch")
    con = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query("SELECT * FROM quakes", con)
    finally:
        con.close()


def _print_side_by_side(baseline: dict, model: dict, title: str) -> None:
    print(f"\n{title}")
    print(f"{'metric':<12}{'baseline':>12}{'model':>12}")
    for key in model:
        print(f"{key:<12}{baseline[key]:>12.4f}{model[key]:>12.4f}")


def _train_regression(args, X_train, X_test, y_train, y_test) -> dict:
    baseline = make_regressor("dummy").fit(X_train, y_train)
    model = make_regressor(args.model, degree=args.degree,
                           alpha=args.alpha).fit(X_train, y_train)

    base_rep = regression_report(y_test, baseline.predict(X_test))
    model_rep = regression_report(y_test, model.predict(X_test))
    _print_side_by_side(base_rep, model_rep, "Regression on test set")

    cv = cross_validate_model(make_regressor(args.model, degree=args.degree,
                                             alpha=args.alpha),
                              X_train, y_train, scoring="r2")
    print(f"\n5-fold CV R2 on train: {cv['mean']:.4f} +/- {cv['std']:.4f}")

    report = {"task": "regression", "model": args.model,
              "degree": args.degree, "alpha": args.alpha,
              "baseline": base_rep, "model_test": model_rep,
              "cv_r2_train": cv}
    return report, model


def _train_classification(args, X_train, X_test, y_train, y_test):
    class_weight = "balanced" if args.balanced else None
    baseline = make_classifier("dummy").fit(X_train, y_train)
    model = make_classifier(args.model, C=args.C,
                            class_weight=class_weight).fit(X_train, y_train)

    # Threshold chosen from out-of-fold predictions on TRAIN only.
    oof = cross_val_predict(
        make_classifier(args.model, C=args.C, class_weight=class_weight),
        X_train, y_train, cv=5, method="predict_proba")[:, 1]
    threshold = best_threshold(y_train, oof, metric="f1")

    base_proba = baseline.predict_proba(X_test)[:, 1]
    proba = model.predict_proba(X_test)[:, 1]

    base_rep = classification_report_dict(
        y_test, (base_proba >= 0.5).astype(int), base_proba)
    default_rep = classification_report_dict(
        y_test, (proba >= 0.5).astype(int), proba)
    tuned_rep = classification_report_dict(
        y_test, (proba >= threshold).astype(int), proba)

    _print_side_by_side(base_rep, default_rep,
                        "Classification on test set (threshold 0.5)")
    _print_side_by_side(base_rep, tuned_rep,
                        f"Classification on test set (threshold {threshold:.2f}"
                        " chosen on train)")

    cv = cross_validate_model(
        make_classifier(args.model, C=args.C, class_weight=class_weight),
        X_train, y_train, scoring="roc_auc")
    print(f"\n5-fold CV ROC-AUC on train: {cv['mean']:.4f} +/- {cv['std']:.4f}")

    report = {"task": "classification", "model": args.model,
              "C": args.C, "balanced": args.balanced,
              "threshold": threshold,
              "baseline": base_rep,
              "model_test_at_0.5": default_rep,
              "model_test_at_threshold": tuned_rep,
              "cv_roc_auc_train": cv}
    return report, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a quake model.")
    parser.add_argument("--task", choices=["regression", "classification"],
                        required=True)
    parser.add_argument("--model", default=None,
                        choices=REGRESSION_MODELS + CLASSIFICATION_MODELS)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()

    if args.model is None:
        args.model = "linear" if args.task == "regression" else "logistic"
    allowed = (REGRESSION_MODELS if args.task == "regression"
               else CLASSIFICATION_MODELS)
    if args.model not in allowed:
        parser.error(f"--model for {args.task} must be one of {allowed}")

    df = load_data()
    print(f"loaded {len(df)} rows from {DB_PATH}")
    X = df.drop(columns=["mag", config.TARGET])
    if args.task == "regression":
        y, stratify = df["mag"], None
    else:
        y = df[config.TARGET]
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=stratify, random_state=config.SEED)

    if args.task == "regression":
        report, model = _train_regression(args, X_train, X_test,
                                          y_train, y_test)
    else:
        report, model = _train_classification(args, X_train, X_test,
                                              y_train, y_test)

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    name = f"{args.task}_{args.model}"
    joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    with open(REPORTS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {MODELS_DIR / (name + '.joblib')}")
    print(f"saved {REPORTS_DIR / (name + '.json')}")


if __name__ == "__main__":
    main()