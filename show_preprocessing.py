import pandas as pd

from src.data_loader import load_raw, validate_raw
from src.preprocess import clean, split, build_preprocessor
from src import config


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def section(title):
    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


def sub_section(title):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


# ============================================================
# STEP 1: LOAD RAW DATA
# ============================================================

section("STEP 1: RAW DATASET - BEFORE PREPROCESSING")

raw = load_raw()

print(f"\nDataset Shape: {raw.shape}")
print(f"Rows    : {raw.shape[0]}")
print(f"Columns : {raw.shape[1]}")

sub_section("FIRST 5 ROWS OF RAW DATA")

print(raw.head().to_string())

sub_section("RAW COLUMN NAMES")

for i, column in enumerate(raw.columns, start=1):
    print(f"{i:2}. {column}")


# ============================================================
# STEP 2: RAW DATA VALIDATION
# ============================================================

section("STEP 2: RAW DATA INSPECTION")

validation = validate_raw(raw)

print("\nDataset Validation Results:")

for key, value in validation.items():
    print(f"{key}: {value}")


# ============================================================
# STEP 3: DATA TYPES BEFORE PREPROCESSING
# ============================================================

section("STEP 3: DATA TYPES BEFORE PREPROCESSING")

print(raw.dtypes.to_string())


# ============================================================
# STEP 4: RAW DATA PROBLEM - BLANK TOTALCHARGES
# ============================================================

section("STEP 4: RAW DATA ISSUE - BLANK TotalCharges")

blank_mask = raw["TotalCharges"].str.strip() == ""

print(f"\nNumber of blank TotalCharges values: {blank_mask.sum()}")

print("\nRows with blank TotalCharges:")

print(
    raw.loc[
        blank_mask,
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ].to_string(index=False)
)

print("\nExplanation:")
print(
    "All blank TotalCharges rows have tenure = 0."
)
print(
    "These are new customers who have not been billed yet."
)
print(
    "Therefore, blank TotalCharges is replaced with 0 instead of using median."
)


# ============================================================
# STEP 5: RAW CATEGORY EXAMPLES
# ============================================================

section("STEP 5: CATEGORICAL VALUES BEFORE CLEANING")

columns_to_show = [
    "MultipleLines",
    "OnlineSecurity",
    "InternetService",
    "PhoneService"
]

for column in columns_to_show:

    print(f"\n{column}:")

    print(
        raw[column]
        .value_counts(dropna=False)
        .to_string()
    )


# ============================================================
# STEP 6: APPLY CLEANING + FEATURE ENGINEERING
# ============================================================

section("STEP 6: APPLYING CLEANING AND FEATURE ENGINEERING")

processed = clean(raw)

print("\nPreprocessing completed successfully.")

print(f"\nShape after preprocessing: {processed.shape}")

print("\nFirst 5 rows after preprocessing:")

print(processed.head().to_string())


# ============================================================
# STEP 7: WHAT CHANGED
# ============================================================

section("STEP 7: WHAT CHANGED AFTER PREPROCESSING")

print("\n1. customerID removed")
print("   Reason: It is only a unique identifier.")

print("\n2. TotalCharges converted to numeric")
print("   Blank values -> 0")

print("\n3. Structural placeholder categories simplified")
print("   'No internet service' -> 'No'")
print("   'No phone service' -> 'No'")

print("\n4. New engineered features added:")
print("   - is_new_customer")
print("   - num_services")
print("   - avg_charge")

print("\n5. Target converted:")
print("   Churn: No -> 0")
print("   Churn: Yes -> 1")


# ============================================================
# STEP 8: SHOW BEFORE VS AFTER
# ============================================================

section("STEP 8: BEFORE VS AFTER PREPROCESSING")

print("\nBEFORE:")
print(f"Shape: {raw.shape}")
print(f"Columns: {len(raw.columns)}")

print("\nAFTER:")
print(f"Shape: {processed.shape}")
print(f"Columns: {len(processed.columns)}")

print("\nRemoved column:")
print("customerID")

print("\nAdded columns:")

new_columns = [
    column
    for column in processed.columns
    if column not in raw.columns
]

for column in new_columns:
    print(f"- {column}")


# ============================================================
# STEP 9: FEATURE ENGINEERING EXAMPLES
# ============================================================

section("STEP 9: FEATURE ENGINEERING EXAMPLES")

columns = [
    "tenure",
    "TotalCharges",
    "MonthlyCharges",
    "is_new_customer",
    "num_services",
    "avg_charge",
    "Churn"
]

print(processed[columns].head(10).to_string(index=False))


# ============================================================
# STEP 10: EXPLAIN FEATURE ENGINEERING
# ============================================================

section("STEP 10: HOW THE NEW FEATURES WERE CREATED")

print("\n1. is_new_customer")
print("   Formula:")
print("   tenure == 0  -> 1")
print("   tenure > 0   -> 0")

print("\n2. num_services")
print("   Counts how many services the customer uses.")

print("\n3. avg_charge")
print("   Formula:")
print("   avg_charge = TotalCharges / max(tenure, 1)")

print("\n   max(tenure, 1) prevents division by zero.")


# ============================================================
# STEP 11: TARGET TRANSFORMATION
# ============================================================

section("STEP 11: TARGET VARIABLE TRANSFORMATION")

print("\nRAW DATA:")

print(raw["Churn"].value_counts().to_string())

print("\nAFTER PREPROCESSING:")

print(processed["Churn"].value_counts().to_string())

print("\nTransformation:")
print("No  -> 0")
print("Yes -> 1")


# ============================================================
# STEP 12: TRAIN TEST SPLIT
# ============================================================

section("STEP 12: STRATIFIED TRAIN / TEST SPLIT")

train_df, test_df = split(processed)

print(f"\nFull dataset : {processed.shape}")
print(f"Training set : {train_df.shape}")
print(f"Testing set  : {test_df.shape}")

print("\nSplit Ratio:")
print("80% -> Training")
print("20% -> Testing")

print("\nWhy Stratified Split?")
print(
    "It preserves approximately the same churn ratio in both training "
    "and testing datasets."
)


# ============================================================
# STEP 13: CHURN DISTRIBUTION
# ============================================================

section("STEP 13: CHURN DISTRIBUTION AFTER SPLIT")

print("\nFULL DATASET:")

print(
    processed["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .rename("Percentage")
    .to_string()
)

print("\nTRAINING DATASET:")

print(
    train_df["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .rename("Percentage")
    .to_string()
)

print("\nTESTING DATASET:")

print(
    test_df["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .rename("Percentage")
    .to_string()
)


# ============================================================
# STEP 14: BEFORE ENCODING
# ============================================================

section("STEP 14: DATA BEFORE ENCODING")

X_train = train_df.drop(columns=[config.TARGET])

print("\nExample categorical columns before encoding:")

example_columns = [
    "Contract",
    "InternetService",
    "PaymentMethod"
]

print(
    X_train[example_columns]
    .head(10)
    .to_string(index=False)
)

print("\nThese are still text categories.")
print("Encoding has NOT happened yet.")


# ============================================================
# STEP 15: PREPROCESSOR CREATION
# ============================================================

section("STEP 15: PREPROCESSING PIPELINE")

print("\nCreating the preprocessing transformer...")

preprocessor = build_preprocessor(scale=False)

print("\nNumeric Columns:")

for column in config.NUMERIC_COLS:
    print(f"- {column}")

print("\nCategorical Columns:")

for column in config.CATEGORICAL_COLS:
    print(f"- {column}")

print("\nImportant:")
print(
    "The preprocessor is created here but NOT fitted on the full dataset."
)
print(
    "It learns category levels and scaling statistics only during model training."
)
print(
    "This prevents data leakage."
)


# ============================================================
# STEP 16: ENCODING DEMONSTRATION
# ============================================================

section("STEP 16: ONE-HOT ENCODING DEMONSTRATION")

print("\nFor demonstration, we fit the preprocessor ONLY on training data.")

X_train_transformed = preprocessor.fit_transform(X_train)

print(f"\nBefore encoding shape: {X_train.shape}")
print(f"After encoding shape : {X_train_transformed.shape}")

print("\nExample:")

print("\nBefore:")

print(
    X_train[
        ["Contract", "InternetService", "PaymentMethod"]
    ]
    .head(5)
    .to_string(index=False)
)

print("\nAfter transformation:")
print(
    "Categorical values become numerical indicator columns using One-Hot Encoding."
)


# ============================================================
# STEP 17: FINAL PIPELINE SUMMARY
# ============================================================

section("FINAL PREPROCESSING PIPELINE SUMMARY")

print(
    """
RAW CSV
   |
   v
1. Load Raw Dataset
   |
   v
2. Validate Dataset
   |
   v
3. Handle blank TotalCharges
   |
   v
4. Collapse redundant categories
   |
   v
5. Feature Engineering
   |
   |---- is_new_customer
   |---- num_services
   |---- avg_charge
   |
   v
6. Convert Target
   |
   |---- No  -> 0
   |---- Yes -> 1
   |
   v
7. Remove customerID
   |
   v
8. Stratified Train/Test Split
   |
   |---- Train: 80%
   |---- Test : 20%
   |
   v
9. Fit Encoder ONLY on Training Data
   |
   v
10. One-Hot Encoding
   |
   v
11. Scaling (when required by the model)
   |
   v
READY FOR MACHINE LEARNING MODEL
"""
)


section("PREPROCESSING DEMONSTRATION COMPLETE")

print("\nYou can now explain the entire preprocessing pipeline step by step.")