# main.py
# The entry point of our Climate Risk API
# Defines the three endpoints users can call
#
# Same structure as fraud detection main.py
# but this time we're predicting climate risk
# instead of transaction fraud

from fastapi import FastAPI
from app.schemas import DisasterInput, RiskOutput, VALID_DISASTER_TYPES, VALID_CONTINENTS
from app.model import predict_risk

# ── CREATE THE APP ────────────────────────────────────────
# FastAPI object with title and description
# these appear automatically on the /docs page

app = FastAPI(
    title="Climate Property Risk API",
    description=(
        "Predicts climate and disaster risk levels for property assessment. "
        "Returns High, Medium or Low risk based on historical disaster patterns. "
        "Built using EM-DAT disaster data (1970-2021)."
    ),
    version="1.0.0"
)

# ── ROOT ENDPOINT ─────────────────────────────────────────
# Simple welcome message
# Visit http://localhost:8000/ to see it

@app.get("/")
def root():
    return {
        "message": "Climate Property Risk API is running",
        "docs": "Visit /docs to test the API",
        "version": "1.0.0"
    }

# ── HEALTH CHECK ──────────────────────────────────────────
# Used by monitoring tools to check if API is alive
# Same pattern as fraud detection

@app.get("/health")
def health():
    return {"status": "healthy"}

# ── VALID OPTIONS ENDPOINT ────────────────────────────────
# New endpoint we didn't have in fraud detection
# Tells users what values are valid for disaster_type
# and continent so they don't have to guess
#
# This is good API design — make it easy for users
# to know what they can send

@app.get("/options")
def options():
    return {
        "valid_disaster_types": VALID_DISASTER_TYPES,
        "valid_continents": VALID_CONTINENTS,
        "note": "For region, use the full region name e.g. 'Western Africa', 'Southern Asia'"
    }

# ── PREDICT ENDPOINT ──────────────────────────────────────
# Main endpoint — send disaster scenario, get risk level back
# POST because we're sending data in the request body

@app.post("/predict", response_model=RiskOutput)
def predict(disaster: DisasterInput):
    return predict_risk(disaster)