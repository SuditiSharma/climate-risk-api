# model.py
# Loads the trained model and encoders
# Takes incoming disaster data and returns a risk prediction
#
# Same structure as fraud detection model.py
# but we have extra steps because our inputs include text
# that needs to be encoded before the model can use it

import joblib
import numpy as np
from app.schemas import DisasterInput, RiskOutput
from app.schemas import VALID_DISASTER_TYPES, VALID_CONTINENTS

# ── LOAD MODEL AND ENCODERS ───────────────────────────────
# We load everything once when the API starts
# not every time a request comes in
# same logic as fraud detection — saves time

print("Loading model and encoders...")

model        = joblib.load("model/climate_model.pkl")
le_disaster  = joblib.load("model/le_disaster.pkl")
le_continent = joblib.load("model/le_continent.pkl")
le_region    = joblib.load("model/le_region.pkl")

print("Model and encoders loaded successfully!")

# ── PREDICTION FUNCTION ───────────────────────────────────
def predict_risk(disaster: DisasterInput) -> RiskOutput:

    # Step 1 — Validate text inputs
    # ──────────────────────────────
    # Check disaster type is something the model knows about
    # if not — return a helpful error message
    # rather than letting the encoder crash

    if disaster.disaster_type not in VALID_DISASTER_TYPES:
        return RiskOutput(
            risk_level="Unknown",
            confidence=0.0,
            message=f"Unknown disaster type '{disaster.disaster_type}'. "
                    f"Valid types: {', '.join(VALID_DISASTER_TYPES)}"
        )

    if disaster.continent not in VALID_CONTINENTS:
        return RiskOutput(
            risk_level="Unknown",
            confidence=0.0,
            message=f"Unknown continent '{disaster.continent}'. "
                    f"Valid continents: {', '.join(VALID_CONTINENTS)}"
        )

    # Step 2 — Encode text to numbers
    # ──────────────────────────────
    # Convert text inputs to numbers using the saved encoders
    # This is the reverse of what we did in train.py
    # train.py: "Flood" → 2
    # here:     "Flood" → encoder → 2

    try:
        disaster_type_enc = le_disaster.transform([disaster.disaster_type])[0]
        continent_enc     = le_continent.transform([disaster.continent])[0]
        # for region we use a try/except because there are many regions
        # and the user might send one that wasn't in training data
        try:
            region_enc = le_region.transform([disaster.region])[0]
        except ValueError:
            # if region is unknown, use 0 as a fallback
            # not ideal but better than crashing
            region_enc = 0

    except Exception as e:
        return RiskOutput(
            risk_level="Error",
            confidence=0.0,
            message=f"Encoding error: {str(e)}"
        )

    # Step 3 — Build feature array
    # ──────────────────────────────
    # Same as fraud detection — arrange into 2D numpy array
    # one row, seven columns
    # order must match exactly what we used in train.py

    features = np.array([[
        disaster_type_enc,
        continent_enc,
        region_enc,
        disaster.total_deaths,
        disaster.people_affected,
        disaster.total_damages,
        disaster.magnitude
    ]])

    # Step 4 — Make prediction
    # ──────────────────────────────
    # predict() returns the risk level — High, Medium or Low
    # predict_proba() returns confidence for each class

    risk_level   = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    # get the confidence for the predicted class
    # model.classes_ tells us the order of classes
    # so we can match probability to the right label
    class_index = list(model.classes_).index(risk_level)
    confidence  = probabilities[class_index]

    # Step 5 — Build human readable message
    # ──────────────────────────────────────
    if risk_level == "High":
        message = (
            f"⚠️ HIGH RISK — {round(confidence * 100, 1)}% confidence. "
            f"This scenario shows patterns consistent with historically "
            f"high-impact disaster events."
        )
    elif risk_level == "Medium":
        message = (
            f"⚡ MEDIUM RISK — {round(confidence * 100, 1)}% confidence. "
            f"This scenario shows moderate risk indicators. "
            f"Monitoring recommended."
        )
    else:
        message = (
            f"✅ LOW RISK — {round(confidence * 100, 1)}% confidence. "
            f"This scenario shows patterns consistent with "
            f"lower-impact historical events."
        )

    # Step 6 — Return the result
    return RiskOutput(
        risk_level=risk_level,
        confidence=round(float(confidence), 4),
        message=message
    )