# ML Experiment Tracking with MLflow

## 1. Project Overview

This project demonstrates how to use MLflow for tracking, comparing, and selecting machine learning experiments.

The task uses the California Housing dataset to train a regression model with three different hyperparameter configurations.

MLflow is used to track:

- Model parameters
- Validation metrics
- Trained model artifacts
- Experiment results

The primary metric used for model selection is RMSE.

---

## 2. Dataset

The California Housing dataset is provided by Scikit-learn.

The dataset contains information about California districts and their corresponding median house values.

The dataset is split into:

- 80% Training data
- 20% Validation data

A fixed random seed of `42` is used to make the split reproducible.

### Dataset Split

| Dataset | Samples |
|---|---:|
| Training | 16,512 |
| Validation | 4,128 |

---

## 3. Machine Learning Model

The model used in all experiments is:

**HistGradientBoostingRegressor**

The same model architecture is trained three times using different values for:

- `max_depth`
- `learning_rate`

The experiments are:

| Run | Max Depth | Learning Rate |
|---|---:|---:|
| Run 1 | 3 | 0.10 |
| Run 2 | 5 | 0.05 |
| Run 3 | 7 | 0.01 |

---

## 4. Experiment Tracking with MLflow

MLflow is used to track each training run.

For every experiment, the following parameters are logged:

- `max_depth`
- `learning_rate`

The following validation metrics are logged:

- RMSE
- MAE
- R²

The trained model is also saved as an MLflow model artifact.

MLflow Experiment:

`California Housing Experiments`

---

## 5. Experiment Results

The three experiments produced the following validation results:

| Run | Max Depth | Learning Rate | RMSE | MAE | R² |
|---|---:|---:|---:|---:|---:|
| Run 1 | 3 | 0.10 | 0.5406 | 0.3702 | 0.7769 |
| **Run 2** | **5** | **0.05** | **0.5222** | **0.3549** | **0.7919** |
| Run 3 | 7 | 0.01 | 0.7213 | 0.5509 | 0.6029 |

### Metric Interpretation

- RMSE: Lower is better.
- MAE: Lower is better.
- R²: Higher is better.

RMSE was used as the primary metric for selecting the best model.

---

## 6. Best Model

The best-performing configuration was:

| Parameter | Value |
|---|---:|
| Max Depth | 5 |
| Learning Rate | 0.05 |
| RMSE | 0.5222 |
| MAE | 0.3549 |
| R² | 0.7919 |

### Why was Run 2 selected?

Run 2 was selected because it achieved the lowest validation RMSE among the three experiments.

Its RMSE was `0.5222`, compared with `0.5406` for Run 1 and `0.7213` for Run 3.

Run 2 also achieved the lowest MAE and the highest R², providing additional evidence that it was the best-performing configuration among the tested experiments.

---

## 7. MLflow UI

The experiments were tracked and compared using the MLflow UI.

The MLflow experiment contains the parameters, metrics, and trained model artifacts for the three runs.


## 8. Project Structure

```text
mlflow-task/
│
├── train.py
├── requirements.txt
└── README.md
