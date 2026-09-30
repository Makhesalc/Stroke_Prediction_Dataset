from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "data" / "stroke_data_cleaned.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

# Scale continuous measurements.
NUMERIC_COLUMNS = ["age", "avg_glucose_level", "bmi"]

# These columns already contain 0 and 1.
BINARY_COLUMNS = ["hypertension", "heart_disease"]

# Convert these categories into numerical columns.
CATEGORICAL_COLUMNS = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]


def build_preprocessor():
    """Create the steps used to prepare patient characteristics."""

    numeric_pipeline = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    binary_pipeline = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
    ])

    categorical_pipeline = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("binary", binary_pipeline, BINARY_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def main():
    # 1. Load the cleaned dataset.
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}\n"
            "Run clean_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("The dataset contains no patient records.")

    # 2. Check that the required input columns exist.
    input_columns = (
        NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS
    )

    missing_columns = set(input_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Selecting only these columns excludes both id and stroke.
    X = df[input_columns].copy()

    for column in NUMERIC_COLUMNS + BINARY_COLUMNS:
        X[column] = pd.to_numeric(X[column], errors="raise")

    empty_columns = X.columns[X.isna().all()].tolist()
    if empty_columns:
        raise ValueError(
            f"These columns have no usable values: {empty_columns}"
        )

    for column in BINARY_COLUMNS:
        if not X[column].dropna().isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0, 1 or missing values.")

    # 3. Fit the preprocessing steps and transform the data.
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    prepared_data = pd.DataFrame(
        transformed,
        columns=preprocessor.get_feature_names_out(),
        index=df.index,
    )

    # 4. Create output folders.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 5. Save the numerical inputs for clustering.
    prepared_data.to_csv(
        OUTPUT_DIR / "clustering_inputs.csv",
        index=False,
    )

    # Save the original records in the SAME row order.
    # Later, cluster labels can be attached to these records.
    df.to_csv(
        OUTPUT_DIR / "patients_for_interpretation.csv",
        index=False,
    )

    # 6. Save the fitted preprocessing setup for future patients.
    joblib.dump(
        preprocessor,
        MODEL_DIR / "preprocessor.pkl",
    )

    print("Preprocessing completed!")
    print(f"Number of patients: {prepared_data.shape[0]}")
    print(f"Number of prepared features: {prepared_data.shape[1]}")

    print("\nFirst five prepared rows:")
    print(prepared_data.head())

    print("\nFiles saved:")
    print(OUTPUT_DIR / "clustering_inputs.csv")
    print(OUTPUT_DIR / "patients_for_interpretation.csv")
    print(MODEL_DIR / "preprocessor.pkl")


if __name__ == "__main__":
    main()