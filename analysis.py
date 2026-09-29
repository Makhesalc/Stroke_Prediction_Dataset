from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "stroke_data_cleaned.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def load_cleaned_data():
    """Load the dataset created by clean_data.py."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found at:\n{DATA_PATH}\n"
            "Run clean_data.py first."
        )

    return pd.read_csv(DATA_PATH)


def display_summary(df):
    """Display basic information about the dataset."""
    print("DATASET SUMMARY")
    print(f"Patients: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nFirst five records:")
    print(df.head())

    print("\nColumn types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nNumerical statistics:")
    print(df.describe().round(2))

    print("\nCategorical value counts:")
    for column in df.select_dtypes(include=["object", "string"]).columns:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False))

    # Stroke is examined for context, not used as a clustering input.
    if "stroke" in df.columns:
        counts = (
            df["stroke"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )

        summary = pd.DataFrame({
            "Patients": counts,
            "Percentage": (counts / len(df) * 100).round(2)
        })
        summary.index = ["No recorded stroke", "Recorded stroke"]

        print("\nRecorded stroke distribution:")
        print(summary)


def save_graph(fig, filename):
    """Save and close a graph."""
    fig.tight_layout()
    destination = OUTPUT_DIR / filename
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph saved: {destination.name}")


def create_graphs(df):
    """Create exploratory graphs for the clustering project."""
    sns.set_theme(style="whitegrid")

    measurements = {
        "age": "Age (years)",
        "bmi": "BMI",
        "avg_glucose_level": "Average glucose level"
    }

    # 1. Distributions of the main numerical characteristics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (column, label) in zip(axes, measurements.items()):
        sns.histplot(data=df, x=column, bins=30, ax=ax)
        ax.set_title(f"{label} distribution")
        ax.set_xlabel(label)
        ax.set_ylabel("Number of patients")

    save_graph(fig, "patient_distributions.png")

    # 2. Box plots to help identify unusual values
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (column, label) in zip(axes, measurements.items()):
        sns.boxplot(data=df, y=column, ax=ax)
        ax.set_title(label)
        ax.set_ylabel(label)

    save_graph(fig, "patient_boxplots.png")

    # 3. Relationship between age and glucose
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="age",
        y="avg_glucose_level",
        alpha=0.4,
        s=25,
        ax=ax
    )
    ax.set_title("Age and average glucose level")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Average glucose level")

    save_graph(fig, "age_vs_glucose.png")

    # 4. Correlations among numerical patient characteristics
    # Exclude identifiers and stroke outcomes.
    numeric_inputs = (
        df.drop(columns=["id", "stroke"], errors="ignore")
        .select_dtypes(include="number")
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        numeric_inputs.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax
    )
    ax.set_title("Correlations between numerical characteristics")

    save_graph(fig, "correlation_heatmap.png")


def main():
    df = load_cleaned_data()

    if df.empty:
        raise ValueError("The cleaned dataset contains no patient records.")

    required_columns = {"age", "bmi", "avg_glucose_level"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    display_summary(df)

    # Save statistics for your project report
    df.describe(include="all").to_csv(
        OUTPUT_DIR / "summary_statistics.csv"
    )

    create_graphs(df)

    print("\nAnalysis completed!")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()