from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATA_PATH = OUTPUT_DIR / "patients_with_clusters.csv"


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"File not found: {DATA_PATH}\n"
            "Run cluster_model.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required = {"cluster", "stroke"}
    if not required.issubset(df.columns):
        raise ValueError("The dataset needs cluster and stroke columns.")

    if not df["stroke"].dropna().isin([0, 1]).all():
        raise ValueError("Stroke values must be 0, 1 or missing.")

    # Only use patients with a known cluster and stroke outcome.
    valid = df.dropna(subset=["cluster", "stroke"])

    if valid.empty:
        raise ValueError("No records have both cluster and stroke values.")

    print(f"Records excluded for missing values: {len(df) - len(valid)}")

    # Rows = patient clusters; columns = recorded stroke status.
    counts = pd.crosstab(
        valid["cluster"],
        valid["stroke"],
    ).reindex(columns=[0, 1], fill_value=0)

    counts.columns = ["No recorded stroke", "Recorded stroke"]

    # Percentages within each cluster.
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    counts.to_csv(OUTPUT_DIR / "cluster_stroke_counts.csv")
    percentages.to_csv(OUTPUT_DIR / "cluster_stroke_percentages.csv")

    print("\nPATIENT COUNTS")
    print(counts.to_string())

    print("\nPERCENTAGES WITHIN EACH CLUSTER")
    print(percentages.round(2).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(
        counts,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
    )
    axes[0].set_title("Recorded stroke status by cluster")
    axes[0].set_xlabel("Recorded outcome")
    axes[0].set_ylabel("Cluster")

    sns.heatmap(
        percentages,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        ax=axes[1],
        cbar_kws={"label": "Percentage (%)"},
    )
    axes[1].set_title("Percentages within each cluster")
    axes[1].set_xlabel("Recorded outcome")
    axes[1].set_ylabel("Cluster")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "cluster_stroke_table.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nSaved: outputs/cluster_stroke_table.png")


if __name__ == "__main__":
    main()