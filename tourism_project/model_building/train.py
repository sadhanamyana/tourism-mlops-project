
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# ---------------------------------------------------------
# 1. File paths
# ---------------------------------------------------------

TRAIN_PATH = "tourism_project/model_building/prepared_data/train.csv"
TEST_PATH = "tourism_project/model_building/prepared_data/test.csv"

MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

TARGET_COLUMN = "ProdTaken"


# ---------------------------------------------------------
# 2. Load train and test data
# ---------------------------------------------------------

print("Loading training and testing data...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")


# ---------------------------------------------------------
# 3. Separate features and target
# ---------------------------------------------------------

X_train = train_df.drop(columns=[TARGET_COLUMN])
y_train = train_df[TARGET_COLUMN]

X_test = test_df.drop(columns=[TARGET_COLUMN])
y_test = test_df[TARGET_COLUMN]

print(f"\nNumber of features: {X_train.shape[1]}")


# ---------------------------------------------------------
# 4. Identify numerical and categorical columns
# ---------------------------------------------------------

categorical_columns = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# ---------------------------------------------------------
# 5. Create preprocessing pipeline
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# ---------------------------------------------------------
# 6. Define model
# ---------------------------------------------------------

model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)


# ---------------------------------------------------------
# 7. Create complete pipeline
# ---------------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ---------------------------------------------------------
# 8. Define hyperparameter grid
# ---------------------------------------------------------

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

print("\nHyperparameter grid:")
for parameter, values in param_grid.items():
    print(f"{parameter}: {values}")


# ---------------------------------------------------------
# 9. Configure MLflow experiment
# ---------------------------------------------------------

mlflow.set_experiment("Tourism_Package_Prediction")


# ---------------------------------------------------------
# 10. Hyperparameter tuning
# ---------------------------------------------------------

print("\nStarting hyperparameter tuning...")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

with mlflow.start_run(run_name="RandomForest_GridSearch"):

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("\nBest parameters:")
    print(best_params)

    print(f"\nBest cross-validation F1 score: "
          f"{grid_search.best_score_:.4f}")


    # -----------------------------------------------------
    # 11. Predictions and evaluation
    # -----------------------------------------------------

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_prob)


    # -----------------------------------------------------
    # 12. Log parameters and metrics to MLflow
    # -----------------------------------------------------

    mlflow.log_params(best_params)

    mlflow.log_metric("cv_f1", grid_search.best_score_)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)
    mlflow.log_metric("test_roc_auc", roc_auc)


    # -----------------------------------------------------
    # 13. Display evaluation results
    # -----------------------------------------------------

    print("\nModel Evaluation")
    print("----------------------------")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))


    # -----------------------------------------------------
    # 14. Log model to MLflow
    # -----------------------------------------------------

    mlflow.sklearn.log_model(
        best_model,
        name="best_model"
    )


# ---------------------------------------------------------
# 15. Save the best model locally
# ---------------------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nBest model saved successfully.")
print(f"Model path: {MODEL_PATH}")

print("\nModel training and registration completed successfully.")
