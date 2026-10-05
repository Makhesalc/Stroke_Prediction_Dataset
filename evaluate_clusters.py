from pathlib import Path
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

INPUT_PATH = OUTPUT_DIR / "clustering_inputs.csv"
PATIENT_PATH = OUTPUT_DIR / "patients_with_clusters.csv"
MODEL_PATH = PROJECT_DIR / "models" / "patient_cluster_model.pkl"


def load_files():
    """Load the prepared inputs, patient groups and trained model."""
    for path in [INPUT_PATH, PATIENT_PATH, MODEL_PATH]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing file: {path}\n"
                "Run preprocess.py and cluster_model.py first."
            )

    X = pd.read_csv(INPUT_PATH)
    patients = pd.read_csv(PATIENT_PATH)
    model = joblib.load(MODEL_PATH)

    if len(X) != len(patients):
        raise ValueError(
            "Patient records and clustering inputs have different lengths. "
            "Run cluster_model.py again."
        )

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Inputs contain missing or infinite values.")

    if "cluster" not in patients.columns:
        raise ValueError("The patient file needs a 'cluster' column.")

    # Check that the saved assignments agree with the current model.
    labels = model.predict(X)

    if not np.array_equal(labels, patients["cluster"].to_numpy()):
        raise ValueError(
            "Saved cluster labels do not match the current model. "
            "Run cluster_model.py again."
        )

    return X, patients, model


def plot_clusters(X, patients):
    """Project the data into two dimensions for visualisation only."""
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(9, 6))

    for cluster in sorted(patients["cluster"].unique()):
        mask = patients["cluster"].to_numpy() == cluster

        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            label=f"Cluster {cluster}",
            alpha=0.45,
            s=18,
        )

    ax.set_title("Patient clusters — PCA visualisation")
    ax.set_xlabel(f"Principal component 1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"Principal component 2 ({explained[1]:.1f}%)")
    ax.legend(title="Patient group")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "clusters_pca.png", dpi=200)
    plt.close(fig)

    print(
        "\nThe PCA graph displays "
        f"{explained.sum():.1f}% of the total input variance."
    )


def compare_cluster_sizes(patients):
    """Count the patients in each cluster."""
    counts = patients["cluster"].value_counts().sort_index()

    summary = pd.DataFrame({
        "patient_count": counts,
        "percentage_of_dataset": counts / len(patients) * 100,
    })
    summary.index.name = "cluster"
    summary.to_csv(OUTPUT_DIR / "cluster_sizes.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index.astype(str), counts.values)

    ax.bar_label(bars)
    ax.set_title("Number of patients in each cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Patient count")
    ax.margins(y=0.15)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cluster_sizes.png", dpi=200)
    plt.close(fig)

    print("\nCLUSTER SIZES")
    print(summary.round(2).to_string())


def compare_stroke_percentages(patients):
    """Compare recorded stroke cases after clustering."""
    if "stroke" not in patients.columns:
        print("\nStroke comparison skipped: no stroke column.")
        return

    stroke = patients["stroke"]

    if not stroke.dropna().isin([0, 1]).all():
        raise ValueError("Stroke values must be 0, 1 or missing.")

    if stroke.notna().sum() == 0:
        print("\nStroke comparison skipped: all outcomes are missing.")
        return

    summary = patients.groupby("cluster").agg(
        total_patients=("stroke", "size"),
        known_outcomes=("stroke", "count"),
        stroke_cases=("stroke", "sum"),
    )

    # Missing outcomes are excluded from the denominator.
    summary["stroke_percentage"] = (
        summary["stroke_cases"]
        / summary["known_outcomes"].replace(0, np.nan)
        * 100
    )

    overall_percentage = stroke.mean() * 100

    summary["difference_from_overall_percentage_points"] = (
        summary["stroke_percentage"] - overall_percentage
    )

    summary.to_csv(OUTPUT_DIR / "stroke_comparison.csv")

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        summary.index.astype(str),
        summary["stroke_percentage"],
    )

    ax.bar_label(
        bars,
        labels=[
            f"{value:.1f}%" if pd.notna(value) else "No outcomes"
            for value in summary["stroke_percentage"]
        ],
    )

    ax.axhline(
        overall_percentage,
        color="red",
        linestyle="--",
        label=f"Overall: {overall_percentage:.1f}%",
    )

    ax.set_title("Recorded stroke percentage by cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Patients with recorded stroke (%)")
    ax.margins(y=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stroke_comparison.png", dpi=200)
    plt.close(fig)

    print("\nRECORDED STROKE COMPARISON")
    print(summary.round(2).to_string())
    print(f"\nOverall recorded stroke percentage: {overall_percentage:.2f}%")


def check_stability(X, model):
    """Compare group assignments from different random initialisations."""
    seeds = [0, 10, 20, 30, 42]
    assignments = {}

    print("\nChecking stability across five random seeds...")

    for seed in seeds:
        # Keep the original model settings, changing only the seed.
        parameters = model.get_params()
        parameters["random_state"] = seed

        repeated_model = KMeans(**parameters)
        assignments[seed] = repeated_model.fit_predict(X)

    comparisons = []

    for seed_a, seed_b in combinations(seeds, 2):
        score = adjusted_rand_score(
            assignments[seed_a],
            assignments[seed_b],
        )

        comparisons.append({
            "seed_a": seed_a,
            "seed_b": seed_b,
            "adjusted_rand_score": score,
        })

    results = pd.DataFrame(comparisons)
    results.to_csv(
        OUTPUT_DIR / "cluster_stability.csv",
        index=False,
    )

    print("\nSTABILITY RESULTS")
    print(results.round(4).to_string(index=False))
    print(
        "\nMean adjusted Rand score: "
        f"{results['adjusted_rand_score'].mean():.4f}"
    )
    print(
        "Lowest adjusted Rand score: "
        f"{results['adjusted_rand_score'].min():.4f}"
    )


def main():
    X, patients, model = load_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {model.n_clusters} clusters...")
    print(f"Patients: {len(patients)}")

    plot_clusters(X, patients)
    compare_cluster_sizes(patients)
    compare_stroke_percentages(patients)
    check_stability(X, model)

    print("\nEvaluation completed!")
    print(f"Graphs and tables saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()