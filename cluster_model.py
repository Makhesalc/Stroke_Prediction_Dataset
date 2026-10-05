from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

INPUT_PATH = OUTPUT_DIR / "clustering_inputs.csv"
PATIENT_PATH = OUTPUT_DIR / "patients_for_interpretation.csv"


def load_data():
    """Load the files created by preprocess.py."""
    for path in [INPUT_PATH, PATIENT_PATH]:
        if not path.is_file():
            raise FileNotFoundError(
                f"File not found: {path}\n"
                "Run preprocess.py first."
            )

    X = pd.read_csv(INPUT_PATH)
    patients = pd.read_csv(PATIENT_PATH)

    if len(X) != len(patients):
        raise ValueError(
            "The input files have different numbers of rows. "
            "Run preprocess.py again."
        )

    if len(X) < 3:
        raise ValueError("At least three patient records are required.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "Clustering inputs contain missing or infinite values."
        )

    return X, patients


def train_models(X):
    """Compare different cluster counts without using stroke outcomes."""
    results = []
    best_model = None
    best_score = -np.inf

    # Avoid requesting more clusters than distinct patient patterns.
    max_k = min(8, len(X) - 1, len(X.drop_duplicates()))

    if max_k < 2:
        raise ValueError("There are not enough distinct patient patterns.")

    for k in range(2, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
        )

        labels = model.fit_predict(X)

        if not 2 <= len(np.unique(labels)) < len(X):
            continue

        score = silhouette_score(X, labels)

        results.append({
            "clusters": k,
            "silhouette_score": score,
            "inertia": model.inertia_,
        })

        print(f"Clusters: {k} | Silhouette score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise ValueError("No valid clustering solution was found.")

    return best_model, pd.DataFrame(results)


def save_comparison_graph(results):
    """Save silhouette and elbow graphs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        results["clusters"],
        results["silhouette_score"],
        marker="o",
    )
    axes[0].set_title("Silhouette scores")
    axes[0].set_xlabel("Number of clusters")
    axes[0].set_ylabel("Silhouette score — higher is better")

    axes[1].plot(
        results["clusters"],
        results["inertia"],
        marker="o",
    )
    axes[1].set_title("Elbow graph")
    axes[1].set_xlabel("Number of clusters")
    axes[1].set_ylabel("Inertia")

    for ax in axes:
        ax.set_xticks(results["clusters"])
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "cluster_comparison.png",
        dpi=200,
    )
    plt.close(fig)


def summarise_clusters(patients, labels):
    """Describe each cluster using the original patient measurements."""
    patients = patients.copy()

    # Both input files must remain in the order saved by preprocess.py.
    patients["cluster"] = labels

    patients.to_csv(
        OUTPUT_DIR / "patients_with_clusters.csv",
        index=False,
    )

    grouped = patients.groupby("cluster")

    summary = grouped.agg(
        patient_count=("age", "size"),
        average_age=("age", "mean"),
        average_bmi=("bmi", "mean"),
        average_glucose=("avg_glucose_level", "mean"),
        hypertension_proportion=("hypertension", "mean"),
        heart_disease_proportion=("heart_disease", "mean"),
    )

    # Examine stroke ONLY after the clusters have been selected.
    if "stroke" in patients.columns:
        summary["recorded_stroke_cases"] = grouped["stroke"].sum()
        summary["known_stroke_records"] = grouped["stroke"].count()
        summary["recorded_stroke_percentage"] = (
            grouped["stroke"].mean() * 100
        )

    summary = summary.round(2)
    summary.to_csv(OUTPUT_DIR / "cluster_summary.csv")

    print("\nCLUSTER SUMMARY")
    print(summary.to_string())


def main():
    X, patients = load_data()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Comparing K-Means models...\n")
    best_model, results = train_models(X)

    results.to_csv(
        OUTPUT_DIR / "cluster_comparison.csv",
        index=False,
    )

    joblib.dump(
        best_model,
        MODEL_DIR / "patient_cluster_model.pkl",
    )

    save_comparison_graph(results)
    summarise_clusters(patients, best_model.labels_)

    print("\nTraining completed!")
    print(f"Selected number of clusters: {best_model.n_clusters}")
    print(
        "Highest silhouette score: "
        f"{results['silhouette_score'].max():.4f}"
    )
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Model saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()