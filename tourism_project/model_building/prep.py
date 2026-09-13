
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = "tourism_project/data/tourism.csv"
OUTPUT_DIR = "tourism_project/model_building/prepared_data"

TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "test.csv")

TARGET_COLUMN = "ProdTaken"


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Original dataset shape: {df.shape}")


# ---------------------------------------------------------
# Remove unnecessary columns
# ---------------------------------------------------------

columns_to_remove = ["Unnamed: 0", "CustomerID"]

# Only remove columns that are actually present
columns_to_remove = [
    column for column in columns_to_remove
    if column in df.columns
]

df = df.drop(columns=columns_to_remove)

print(f"\nRemoved columns: {columns_to_remove}")
print(f"Cleaned dataset shape: {df.shape}")


# ---------------------------------------------------------
# Clean inconsistent categorical values
# ---------------------------------------------------------

if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({
        "Fe Male": "Female"
    })

print("\nGender values after cleaning:")
print(df["Gender"].value_counts())


# ---------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


# ---------------------------------------------------------
# Train-Test Split
# ---------------------------------------------------------

print("\nCreating train-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# Recombine features and target
# ---------------------------------------------------------

train_data = X_train.copy()
train_data[TARGET_COLUMN] = y_train

test_data = X_test.copy()
test_data[TARGET_COLUMN] = y_test


# ---------------------------------------------------------
# Save train and test datasets
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_data.to_csv(TRAIN_PATH, index=False)
test_data.to_csv(TEST_PATH, index=False)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\nData preparation completed successfully.")

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

print("\nTraining target distribution:")
print(train_data[TARGET_COLUMN].value_counts())
print(
    train_data[TARGET_COLUMN]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nTesting target distribution:")
print(test_data[TARGET_COLUMN].value_counts())
print(
    test_data[TARGET_COLUMN]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(f"\nTraining data saved to: {TRAIN_PATH}")
print(f"Testing data saved to: {TEST_PATH}")
