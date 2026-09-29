from pathlib import Path
import pandas as pd

from load_data import load_data

PROJECT_DIR = Path(__file__).resolve().parent
CLEANED_PATH = PROJECT_DIR / "data" / "stroke_data_cleaned.csv"


def clean_data(df):
    """Clean the dataset without changing the original DataFrame."""
    cleaned = df.copy()

    # 1. Remove spaces around column names
    cleaned.columns = cleaned.columns.str.strip()

    # 2. Remove spaces around text values
    text_columns = cleaned.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        cleaned[column] = cleaned[column].str.strip()

    # Treat empty cells as missing values
    cleaned = cleaned.replace(r"^\s*$", pd.NA, regex=True)

    # 3. Remove exact duplicate records before dropping patient IDs
    original_rows = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    print(f"Duplicate records removed: {original_rows - len(cleaned)}")

    # 4. Remove the ID column because it is not a patient characteristic
    cleaned = cleaned.drop(columns=["id"], errors="ignore")

    # 5. Ensure BMI values are numeric
    if "bmi" not in cleaned.columns:
        raise ValueError("The dataset must contain a 'bmi' column.")

    cleaned["bmi"] = pd.to_numeric(cleaned["bmi"], errors="coerce")

    # 6. Fill missing BMI values with the median
    median_bmi = cleaned["bmi"].median()

    if pd.isna(median_bmi):
        raise ValueError("Cannot fill BMI: no valid BMI values were found.")

    missing_bmi = cleaned["bmi"].isna().sum()
    cleaned["bmi"] = cleaned["bmi"].fillna(median_bmi)

    print(f"Missing BMI values filled: {missing_bmi}")
    print(f"Median BMI used: {median_bmi:.2f}")

    return cleaned.reset_index(drop=True)


if __name__ == "__main__":
    data = load_data()
    cleaned_data = clean_data(data)

    CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_data.to_csv(CLEANED_PATH, index=False)

    print("\nCleaning completed!")
    print(f"Cleaned dataset shape: {cleaned_data.shape}")

    print("\nRemaining missing values:")
    print(cleaned_data.isna().sum())

    print("\nFirst 5 cleaned rows:")
    print(cleaned_data.head())

    print(f"\nSaved to: {CLEANED_PATH}")