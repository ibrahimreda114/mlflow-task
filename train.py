import mlflow
import mlflow.sklearn

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 1. Load California Housing Dataset
# ============================================================

data = fetch_california_housing()

X = data.data
y = data.target


# ============================================================
# 2. Split Dataset
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")


# ============================================================
# 3. Configure MLflow
# ============================================================

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("California Housing Experiments")


# ============================================================
# 4. Define Experiments
# ============================================================

experiments = [
    {
        "run_name": "Run 1",
        "max_depth": 3,
        "learning_rate": 0.1
    },
    {
        "run_name": "Run 2",
        "max_depth": 5,
        "learning_rate": 0.05
    },
    {
        "run_name": "Run 3",
        "max_depth": 7,
        "learning_rate": 0.01
    }
]


# ============================================================
# 5. Train and Track Experiments
# ============================================================

for experiment in experiments:

    with mlflow.start_run(run_name=experiment["run_name"]):

        max_depth = experiment["max_depth"]
        learning_rate = experiment["learning_rate"]

        print("\n" + "=" * 60)
        print(f"Starting {experiment['run_name']}")
        print(f"max_depth = {max_depth}")
        print(f"learning_rate = {learning_rate}")

        # ----------------------------------------------------
        # Create Model
        # ----------------------------------------------------

        model = HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )

        # ----------------------------------------------------
        # Train Model
        # ----------------------------------------------------

        model.fit(X_train, y_train)

        # ----------------------------------------------------
        # Validation Predictions
        # ----------------------------------------------------

        predictions = model.predict(X_val)

        # ----------------------------------------------------
        # Calculate Metrics
        # ----------------------------------------------------

        rmse = mean_squared_error(
            y_val,
            predictions
        ) ** 0.5

        mae = mean_absolute_error(
            y_val,
            predictions
        )

        r2 = r2_score(
            y_val,
            predictions
        )

        # ----------------------------------------------------
        # Log Parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "max_depth",
            max_depth
        )

        mlflow.log_param(
            "learning_rate",
            learning_rate
        )

        # ----------------------------------------------------
        # Log Metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "RMSE",
            rmse
        )

        mlflow.log_metric(
            "MAE",
            mae
        )

        mlflow.log_metric(
            "R2",
            r2
        )

        # ----------------------------------------------------
        # Log Model Artifact
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=[
        "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
    ]
)

        # ----------------------------------------------------
        # Display Results
        # ----------------------------------------------------

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R2:   {r2:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

        print("=" * 60)


print("\nAll experiments completed successfully.")