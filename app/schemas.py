# schemas.py
# Defines the shape of data coming into and out of our API
# Think of it as the form someone fills in to get a risk prediction
#
# Same concept as fraud detection schemas
# but this time our inputs are a mix of text and numbers
# because we need disaster type, continent, region

from pydantic import BaseModel

# ── INPUT SCHEMA ──────────────────────────────────────────
# This is what someone sends to our /predict endpoint
# They describe a disaster scenario and we assess the risk
#
# Why text for disaster_type, continent, region?
# Because we want the API to be human readable
# Someone sends "Flood" not "2"
# The API handles the encoding internally

class DisasterInput(BaseModel):
    disaster_type: str      # e.g. "Flood", "Storm", "Earthquake"
    continent: str          # e.g. "Africa", "Asia", "Americas"
    region: str             # e.g. "Western Africa", "Southern Asia"
    total_deaths: float     # number of deaths
    people_affected: float  # number of people affected
    total_damages: float    # economic damage in thousands USD
    magnitude: float        # physical magnitude of the event

# ── OUTPUT SCHEMA ─────────────────────────────────────────
# This is what our API sends back after prediction
# Three pieces of information:
# 1. The risk level — High, Medium or Low
# 2. The probability — how confident the model is
# 3. A human readable message explaining the result

class RiskOutput(BaseModel):
    risk_level: str              # "High", "Medium" or "Low"
    confidence: float            # e.g. 0.92 means 92% confident
    message: str                 # human readable explanation

# ── VALID OPTIONS ─────────────────────────────────────────
# These are the exact values the API accepts
# We use these in model.py to validate input
# before passing to the encoder
#
# If someone sends "flood" instead of "Flood"
# or "africa" instead of "Africa"
# we can catch it and return a helpful error

VALID_DISASTER_TYPES = [
    "Drought",
    "Earthquake",
    "Epidemic",
    "Extreme temperature",
    "Flood",
    "Insect infestation",
    "Landslide",
    "Mass movement (dry)",
    "Storm",
    "Volcanic activity",
    "Wildfire"
]

VALID_CONTINENTS = [
    "Africa",
    "Americas",
    "Asia",
    "Europe",
    "Oceania"
]