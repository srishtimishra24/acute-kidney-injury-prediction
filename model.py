import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def resolve_path(docker_path, local_path):
    """
    Use the Docker-mounted path if it exists, otherwise fall back to
    the local file path. This allows the same code to run both locally
    and in the automated marking environment.
    """
    return docker_path if os.path.exists(docker_path) else local_path


# Input and output file locations
TRAIN_PATH = resolve_path("/data/training.csv", "training.csv")
TEST_PATH = resolve_path("/data/test.csv", "test.csv")
OUTPUT_PATH = resolve_path("/data/aki.csv", "aki.csv")

# Probability threshold used to convert model outputs into labels.
# A lower threshold is chosen to prioritise recall, which aligns
# with the F3 evaluation metric used in this coursework.
THRESHOLD = 0.30


def extract_features(df, training=True):
    """
    Convert each patient record into a fixed-length feature vector.

    Creatinine measurements are identified dynamically based on column
    names to avoid making assumptions about how many historical values
    are present in the dataset.
    """
    X_rows = []
    y = []

    for _, row in df.iterrows():
        # Identify all creatinine columns available for this dataset
        creat_cols = [c for c in row.index if c.startswith("creatinine_result_")]
        creats = row[creat_cols].dropna().values

        # Skip records with no valid creatinine history
        if len(creats) == 0:
            continue

        # Use the most recent measurement as the current value
        latest = creats[-1]

        # Estimate baseline from previous measurements
        baseline = np.median(creats[:-1]) if len(creats) > 1 else creats[0]

        # Construct feature set
        X_rows.append({
            "age": row["age"],
            "sex": 1 if row["sex"] == "m" else 0,
            "latest": latest,
            "baseline": baseline,
            "abs_change": latest - baseline,
            "rel_change": (latest - baseline) / baseline if baseline > 0 else 0.0,
            "mean": np.mean(creats),
            "max": np.max(creats),
            "std": np.std(creats),
        })

        # Target label is only available during training
        if training:
            y.append(1 if row["aki"] == "y" else 0)

    X = pd.DataFrame(X_rows)
    return (X, np.array(y)) if training else X


def main():
    # Model training
    train_df = pd.read_csv(TRAIN_PATH)
    X_train, y_train = extract_features(train_df, training=True)

    # Random Forest is used for its robustness on tabular data
    # and its ability to capture non-linear relationships.
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight={0: 1, 1: 4},
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Inference on test data
    test_df = pd.read_csv(TEST_PATH)
    X_test = extract_features(test_df, training=False)

    # Convert predicted probabilities into binary labels
    probs = model.predict_proba(X_test)[:, 1]
    preds = ["y" if p >= THRESHOLD else "n" for p in probs]

    # Output predictions
    pd.DataFrame({"aki": preds}).to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
