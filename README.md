# SWEMLS Coursework 1 – Acute Kidney Injury Prediction

This repository contains the submission for **SWEMLS Coursework 1**, which focuses on building a software system to predict the presence of **acute kidney injury (AKI)** from patient blood test data.

The system trains a model on historical patient data and generates predictions for an unseen dataset in accordance with the provided specification and automated evaluation framework.

---

## Problem Description

Acute kidney injury is a common indicator of patient deterioration and is typically detected through elevated creatinine levels in blood tests. The goal of this coursework is to develop a model that predicts whether AKI is present based on patient demographics and historical creatinine measurements.

The primary evaluation metric is the **F3 score**, which places greater emphasis on recall than precision. This reflects the clinical requirement to minimise false negatives, where a deteriorating patient might otherwise be missed.

---

## Repository Structure


.
├── model.py # Submission entrypoint; trains model and generates aki.csv

├── requirements.txt # Python dependencies

├── Dockerfile # Provided container configuration (unchanged)

├── README.md


Training data, test data, and generated outputs are intentionally excluded from the repository. These files are supplied to the program by the automated testing environment at runtime.

---

## Model Overview

- **Model type**: Logistic Regression  
- **Input features**:
  - Patient age
  - Patient sex
  - Latest creatinine measurement
  - Estimated baseline creatinine
  - Absolute and relative change from baseline
  - Mean and maximum historical creatinine values
- **Learning objective**: Binary classification (`AKI present` / `AKI not present`)
- **Optimisation focus**: Recall, aligned with the F3 score

Class weighting is applied during training to reduce the likelihood of false negatives. A fixed decision threshold is used at inference time to further prioritise recall.

The model is intentionally simple and interpretable, making it suitable for tabular clinical data and consistent with production deployment considerations.

---

## Running the Model

The submission entrypoint is the `model.py` script. When executed, it performs the following steps:

1. Loads the training dataset
2. Trains the classification model
3. Applies the trained model to the test dataset
4. Writes predictions to `aki.csv`
