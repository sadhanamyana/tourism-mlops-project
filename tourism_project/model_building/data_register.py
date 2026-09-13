
import pandas as pd

# Path to the dataset inside the repository
DATA_PATH = "tourism_project/data/tourism.csv"

# Expected columns
EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]

# Load dataset
df = pd.read_csv(DATA_PATH)

# Validate expected columns
missing_columns = [
    column for column in EXPECTED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset validation failed. Missing columns: {missing_columns}"
    )

print("Dataset validation passed successfully.")

# Short dataset summary
print("\nDataset Summary")
print("----------------------------")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nColumn Names:")
print(list(df.columns))

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nTarget Distribution:")
print(df["ProdTaken"].value_counts())

print("\nData registration completed successfully.")
