from pathlib import Path
import pandas as pd


# Get the folder where this Python file is located
PROJECT_DIR = Path(__file__).resolve().parent

# Path to the CSV dataset
DATA_PATH = PROJECT_DIR / "data" / "healthcare-dataset-stroke-data.csv"


def load_data():
    """Load and return the stroke dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}\n\n"
            "Place healthcare-dataset-stroke-data.csv inside the data folder."
        )

    data = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!")
    print(f"Number of entries: {data.shape[0]}")
    print(f"Number of attributes: {data.shape[1]}")

    return data
if __name__ == "__main__":
    df = load_data()



    print("\nFirst five entries:")
    print(df.head())


    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nDuplicate entries:")
    print(df.duplicated().sum())

    print("\nMissing values:")
    print(df.isnull().sum())