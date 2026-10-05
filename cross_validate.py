from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold

from load_data import load_data
from preprocess import (
    build_preprocessor,
    NUMERIC_COLUMNS,
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
)


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

N_FOLDS = 5
CLUSTER_COUNTS = range(2, 11)


def prepare_original_data():
    """Use original records, before any full-dataset preprocessing."""
    df = load_data().copy()

    df.columns = df.columns.str.strip()

    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        df[column] = df[column].str.strip()

    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Remove exact duplicate records before excluding patient IDs.
    df = df.drop_duplicates()

    input_columns = (
        NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS
    )

    missing_columns = set(input_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Follow the chosen cleaning method: remove missing BMI rows.
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    rows_before = len(df)
    df = df.dropna(subset=["bmi"]).copy()

    print(f"Rows removed for missing BMI: {rows_before - len(df)}")

    # Explicit selection excludes both id and stroke.
    X = df[input_columns].copy()

    for column in NUMERIC_COLUMNS + BINARY_COLUMNS:
        X[column] = pd.to_numeric(X[column], errors="raise")

        if np.isinf(X[column].to_numpy(dtype=float)).any():
            raise ValueError(f"{column} contains infinite values.")

    for column in BINARY_COLUMNS:
        if not X[column].dropna().isin([0, 1]).all():
            raise ValueError(f"{column} must contain 0, 1 or missing values.")

    return X.reset_index(drop=True)


def run_cross_validation(X):
    splitter = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=42,
    )

    results = []

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(X), start=1
    ):
        X_train = X.iloc[train_indices]
        X_validation = X.iloc[validation_indices]

        empty_columns = X_train.columns[
            X_train.isna().all()
        ].tolist()

        if empty_columns:
            raise ValueError(
                f"Fold {fold} has empty training columns: {empty_columns}"
            )

        # Create and fit a NEW preprocessor for this fold.
        preprocessor = build_preprocessor()

        train_prepared = preprocessor.fit_transform(X_train)

        # Validation uses the training fold's fitted transformations.
        validation_prepared = preprocessor.transform(X_validation)

        distinct_training_rows = np.unique(
            train_prepared, axis=0
        ).shape[0]

        print(f"\nFOLD {fold} OF {N_FOLDS}")

        for k in CLUSTER_COUNTS:
            score = np.nan
            occupied_clusters = 0

            if k > distinct_training_rows:
                status = "Not enough distinct training records"
            else:
                model = KMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=20,
                )

                model.fit(train_prepared)

                # Assign validation patients to the learned centres.
                labels = model.predict(validation_prepared)
                occupied_clusters = len(np.unique(labels))

                # Silhouette requires at least two occupied groups
                # and fewer groups than validation patients.
                if 2 <= occupied_clusters < len(validation_prepared):
                    score = silhouette_score(
                        validation_prepared,
                        labels,
                    )
                    status = "OK"
                else:
                    status = "Silhouette undefined"

            results.append({
                "fold": fold,
                "clusters": k,
                "training_patients": len(X_train),
                "validation_patients": len(X_validation),
                "occupied_validation_clusters": occupied_clusters,
                "validation_silhouette": score,
                "status": status,
            })

            if pd.notna(score):
                print(f"k={k}: validation silhouette = {score:.4f}")
            else:
                print(f"k={k}: {status}")

    return pd.DataFrame(results)


def main():
    X = prepare_original_data()

    if len(X) < N_FOLDS * 3:
        raise ValueError("Not enough patient records for this evaluation.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = run_cross_validation(X)

    summary = results.groupby("clusters").agg(
        mean_silhouette=("validation_silhouette", "mean"),
        std_silhouette=("validation_silhouette", "std"),
        valid_folds=("validation_silhouette", "count"),
    )

    results.to_csv(
        OUTPUT_DIR / "cross_validation_results.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "cross_validation_summary.csv"
    )

    print("\nCROSS-VALIDATION SUMMARY")
    print(summary.round(4).to_string())

    # Compare only candidates with a valid score in every fold.
    eligible = summary[summary["valid_folds"] == N_FOLDS]

    if eligible.empty:
        print("\nNo cluster count had valid scores in all five folds.")
    else:
        best_k = int(eligible["mean_silhouette"].idxmax())
        best = eligible.loc[best_k]

        print(f"\nSuggested number of clusters: {best_k}")
        print(f"Mean validation silhouette: {best['mean_silhouette']:.4f}")
        print(f"Standard deviation: {best['std_silhouette']:.4f}")

    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()