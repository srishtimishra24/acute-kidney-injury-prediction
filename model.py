import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def resolve_path(docker_path, local_path):
    """
    Return the Docker-mounted path if it exists, otherwise fall back
    to the local filesystem path.
    """
    return docker_path if os.path.exists(docker_path) else local_path


# Input and output locations
TRAIN_PATH = resolve_path("/data/training.csv", "training.csv")
TEST_PATH = resolve_path("/data/test.csv", "test.csv")
OUTPUT_PATH = resolve_path("/data/aki.csv", "aki.csv")

# Historical creatinine measurements provided in the dataset
CREAT_COLS = [f"creatinine_result_{i}" for i in range(48)]

# Classification threshold chosen to favour recall
THRESHOLD = 0.30


def extract_features(df, training=True):
    """
    Convert raw patient records into a feature matrix suitable for
    tabular classification models.
    """
    X_rows = []
    y = []

    for _, row in df.iterrows():
        creats = row[CREAT_COLS].dropna().values
        if len(creats) == 0:
            continue

        latest = creats[-1]
        baseline = np.median(creats[:-1]) if len(creats) > 1 else creats[0]

        X_rows.append({
            "age": row["age"],
            "sex": 1 if row["sex"] == "m" else 0,
            "latest": latest,
            "baseline": baseline,
            "abs_change": latest - baseline,
            "rel_change": (latest - baseline) / baseline if baseline > 0 else 0.0,
            "mean": np.mean(creats),
            "max": np.max(creats),
        })

        if training:
            y.append(1 if row["aki"] == "y" else 0)

    X = pd.DataFrame(X_rows)
    return (X, np.array(y)) if training else X


def main():
    # Train model on labelled data
    train_df = pd.read_csv(TRAIN_PATH)
    X_train, y_train = extract_features(train_df, training=True)

    model = LogisticRegression(
        class_weight={0: 1, 1: 4},
        max_iter=1000
    )
    model.fit(X_train, y_train)

    # Generate predictions for unseen test data
    test_df = pd.read_csv(TEST_PATH)
    X_test = extract_features(test_df, training=False)

    probs = model.predict_proba(X_test)[:, 1]
    preds = ["y" if p >= THRESHOLD else "n" for p in probs]

    # Write predictions in the required format
    pd.DataFrame({"aki": preds}).to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
