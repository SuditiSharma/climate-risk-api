# train.py
# This script loads disaster data, creates risk labels,
# trains a model and saves it
# I built this to predict climate property risk levels
# inspired by my fraud detection project

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os


# ── 1. LOAD DATA ──────────────────────────────────────────
print("Loading disaster data...")
df = pd.read_csv("data/1970-2021_DISASTERS.xlsx - emdat data.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ── 2. CLEAN DATA ─────────────────────────────────────────
# I only want natural disasters, not technological ones
# Also dropping rows where key columns are missing
print("\nCleaning data...")

# Keep only natural disasters
df = df[df["Disaster Group"] == "Natural"]

# Only keep columns we actually need
cols_we_need = [
    "Disaster Type",
    "Continent",
    "Region",
    "Total Deaths",
    "No Affected",
    "Total Damages ('000 US$)",
    "Dis Mag Value"
]
df = df[cols_we_need].copy()

# Rename columns to simpler names — easier to work with
df.columns = [
    "disaster_type",
    "continent",
    "region",
    "total_deaths",
    "people_affected",
    "total_damages",
    "magnitude"
]

# Fill missing numbers with 0
# Missing means it wasn't recorded, not that it didn't happen
# but 0 is a reasonable assumption for our model
numeric_cols = ["total_deaths", "people_affected", "total_damages", "magnitude"]
df[numeric_cols] = df[numeric_cols].fillna(0)

# Drop rows where disaster type, continent or region is missing
# We need these for prediction
df = df.dropna(subset=["disaster_type", "continent", "region"])

print(f"After cleaning: {df.shape}")
print(f"Disaster types: {df['disaster_type'].unique()}")

# ── 3. CREATE RISK LABEL ──────────────────────────────────
# This is the key step — we don't have a risk label
# so we create one from deaths + affected + damages
# Think of it like a doctor calculating a risk score
# from multiple health indicators

print("\nCreating risk labels...")

# Normalise each metric to 0-1 scale
# so one metric doesn't dominate the others
# e.g. damages are in thousands of dollars so very large numbers
def normalise(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return series * 0
    return (series - min_val) / (max_val - min_val)

df["deaths_norm"] = normalise(df["total_deaths"])
df["affected_norm"] = normalise(df["people_affected"])
df["damages_norm"] = normalise(df["total_damages"])
df["magnitude_norm"] = normalise(df["magnitude"])

# Combined risk score — weighted average
# I give deaths more weight because loss of life
# is the most serious outcome
df["risk_score"] = (
    df["deaths_norm"] * 0.4 +       # 40% weight on deaths
    df["affected_norm"] * 0.3 +     # 30% weight on people affected
    df["damages_norm"] * 0.2 +      # 20% weight on damages
    df["magnitude_norm"] * 0.1      # 10% weight on magnitude
)

# Convert score to label
# top 25% = High risk
# bottom 25% = Low risk
# middle 50% = Medium risk
high_threshold = df["risk_score"].quantile(0.75)
low_threshold = df["risk_score"].quantile(0.25)

def assign_risk(score):
    if score >= high_threshold:
        return "High"
    elif score <= low_threshold:
        return "Low"
    else:
        return "Medium"

df["risk_level"] = df["risk_score"].apply(assign_risk)

print(f"Risk distribution:")
print(df["risk_level"].value_counts())

# ── 4. ENCODE CATEGORICAL COLUMNS ─────────────────────────
# ML models only understand numbers, not text
# So we convert disaster_type, continent, region to numbers
# LabelEncoder turns "Flood" → 0, "Storm" → 1 etc
# Example:
# "Flood"      → 2
# "Storm"      → 5
# "Earthquake" → 1
#
# We have to save the encoders alongside the model
# because the API needs to use the same mapping
# when it receives new requests
#
# If training says "Flood" = 2
# the API must also say "Flood" = 2
# not "Flood" = 7 or some other number

print("\nEncoding categorical columns...")

# create one encoder for each text column
# each encoder learns the mapping for its own column

le_disaster = LabelEncoder()
le_continent = LabelEncoder()
le_region = LabelEncoder()

# fit_transform does two things in one step:
# fit   = learn all the unique values and assign numbers
# transform = actually convert the column to numbers
df["disaster_type_enc"] = le_disaster.fit_transform(df["disaster_type"])
df["continent_enc"] = le_continent.fit_transform(df["continent"])
df["region_enc"] = le_region.fit_transform(df["region"])

# Let's see what the encoding looks like
print(f"\nDisaster types and their codes:")
for label, code in zip(le_disaster.classes_,
                       range(len(le_disaster.classes_))):
    print(f"  {label} → {code}")

print(f"\nContinents and their codes:")
for label, code in zip(le_continent.classes_,
                       range(len(le_continent.classes_))):
    print(f"  {label} → {code}")

# Save encoders to disk
# We'll need them in the API to convert incoming text
# to the same numbers the model was trained on
os.makedirs("model", exist_ok=True)
joblib.dump(le_disaster, "model/le_disaster.pkl")
joblib.dump(le_continent, "model/le_continent.pkl")
joblib.dump(le_region, "model/le_region.pkl")

print(f"Disaster types encoded: {list(le_disaster.classes_)}")

# ── 5. PREPARE FEATURES ───────────────────────────────────
# X = the features the model learns from (the questions)
# y = the label we want to predict (risk level)(the answer)
#
# We use the encoded versions of text columns
# and the raw numbers for numeric columns
# We do NOT include the normalised columns we created
# for the risk score — those were just for creating the label
# the model should learn from the raw features
print("\nPreparing features...")

feature_cols = [
    "disaster_type_enc",  # what type of disaster
    "continent_enc",      # which continent
    "region_enc",         # which region
    "total_deaths",       # how many died
    "people_affected",    # how many affected
    "total_damages",      # economic damage
    "magnitude"           # physical magnitude
]

X = df[feature_cols]
y = df["risk_level"]

print(f"Features shape: {X.shape}")
print(f"Label distribution:\n{y.value_counts()}")

# Split into train and test
# 80% for training, 20% for testing
# stratify=y makes sure High/Medium/Low ratio
# is the same in both train and test sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # keep same risk distribution in both splits
)

print(f"\nTraining set: {X_train.shape[0]} records")
print(f"Testing set: {X_test.shape[0]} records")

# ── 6. TRAIN MODEL ────────────────────────────────────────
print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,   # 100 decision trees
    random_state=42,    # same result every run
    n_jobs=-1           # use all CPU cores
)

# .fit() is where the actual learning happens
# the model looks at X_train and y_train
# and figures out the patterns itself
model.fit(X_train, y_train)
print("Training complete!")

# ── 7. EVALUATE MODEL ─────────────────────────────────────
# We test on X_test which the model has NEVER seen
# This tells us how well it generalises to new data
print("\nEvaluating model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Also show which features matter most
# Random Forest can tell us which columns
# were most useful for making predictions
print("\nFeature importance (which columns matter most):")
importances = model.feature_importances_
for feature, importance in zip(feature_cols, importances):
    bar = "█" * int(importance * 50)  # visual bar
    print(f"  {feature:<25} {bar} {round(importance * 100, 1)}%")

# ── 8. SAVE MODEL ─────────────────────────────────────────
# Save the trained model to a file
# The API will load this file when it starts up
# so it doesn't need to retrain every time
joblib.dump(model, "model/climate_model.pkl")
print("\nModel saved to model/climate_model.pkl")
print("Encoders saved to model/")
print("\nAll done — ready to build the API!")