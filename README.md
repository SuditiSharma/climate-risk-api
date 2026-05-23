# 🌍 Climate Property Risk API

A machine learning API that predicts whether a disaster scenario 
represents High, Medium, or Low property risk — built as part of 
my exploration into climate risk assessment methods.

Inspired by the challenge of property risk under climate change, 
and my background working at Aviva (a FTSE 100 insurer) where 
understanding risk from real-world data was part of daily work.

---

## 💡 Why I Built This

I've been reading about AI-powered property risk assessment — 
particularly how insurers and policymakers struggle with fragmented 
disaster data and uncertain risk estimates. I wanted to explore 
this problem hands-on before going deeper into the research.

The interesting challenge here wasn't the model itself — it was 
that the dataset had no risk label. I had to create one from 
scratch using a weighted combination of disaster outcomes. That 
design decision — what to weight, and why — turned out to be the 
most important part of the project.

---

## 🔍 The Problem

The EM-DAT dataset records 16,000+ natural disasters from 1970 
to 2021. It tells you what happened — deaths, damage, affected 
people — but it doesn't tell you the risk level.

So the first challenge was: how do you define risk from outcomes?

I built a weighted risk score:

| Indicator | Weight | Reasoning |
|---|---|---|
| Total deaths | 40% | Loss of life is the most serious outcome |
| People affected | 30% | Humanitarian scale matters |
| Economic damages | 20% | Property and infrastructure impact |
| Physical magnitude | 10% | Scale of the event itself |

Then I used percentile thresholds to create three balanced 
categories — top 25% = High, bottom 25% = Low, middle 50% = Medium.

Is this the perfect definition of risk? No. But it's a 
defensible starting point — and understanding its limitations 
is part of the point.

---

## 📊 What the Data Showed

After training, I looked at feature importance to understand 
what the model actually learned:
total_deaths        ██████████████████  30.4%
people_affected     ████████████████    27.6%
total_damages       ██████████████      24.7%
magnitude           ████████            13.2%
region              █                   1.2%
continent           █                   1.1%
disaster_type       █                   1.8%
The model learned that risk is almost entirely determined by 
outcomes — not by where the disaster happened or what type it was.

This is an interesting finding: a flood in a dense urban area 
is far more dangerous than a larger flood in a remote region. 
Geography matters — but it matters through context (population 
density, infrastructure) that this dataset doesn't capture.

That gap is exactly what more sophisticated research — combining 
satellite data, population data, and building stock information — 
would address.

---

## 🛠️ Tech Stack

- **Python 3.11**
- **FastAPI** — REST API framework
- **scikit-learn** — Random Forest classifier
- **pandas / numpy** — data processing and feature engineering
- **joblib** — model and encoder serialisation
- **uvicorn** — ASGI server

---

## 📁 Project Structure
climate-risk-api/
├── app/
│   ├── init.py
│   ├── main.py        # API endpoints
│   ├── model.py       # prediction logic
│   └── schemas.py     # input/output validation
├── data/              # EM-DAT dataset (not tracked)
├── model/             # saved model and encoders
├── train.py           # model training script
├── requirements.txt
└── README.md

---

## ⚙️ Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/SuditiSharma/climate-risk-api.git
cd climate-risk-api
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add the dataset**

Download from [EM-DAT via Kaggle](https://www.kaggle.com/datasets/brsdincer/all-natural-disasters-19002021-eosdis) 
and place in the `data/` folder.

**5. Train the model**
```bash
python train.py
```

**6. Run the API**
```bash
uvicorn app.main:app --reload
```

**7. Open docs**
http://127.0.0.1:8000/docs

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Welcome message |
| `/health` | GET | Health check |
| `/options` | GET | Lists valid input values |
| `/predict` | POST | Returns risk prediction |

---

## 📬 Example Request

```json
POST /predict
{
  "disaster_type": "Flood",
  "continent": "Africa",
  "region": "Western Africa",
  "total_deaths": 5000,
  "people_affected": 2000000,
  "total_damages": 50000,
  "magnitude": 0
}
```

## ✅ Example Response

```json
{
  "risk_level": "High",
  "confidence": 1.0,
  "message": "⚠️ HIGH RISK — 100.0% confidence. This scenario 
  shows patterns consistent with historically high-impact 
  disaster events."
}
```

---

## 📊 Model Performance
              precision  recall  f1-score
High            0.99      0.98    0.98
Low             0.99      1.00    0.99
Medium          0.99      0.99    0.99
Overall accuracy: 99%
The high accuracy reflects that the model learned the weighted 
combination used to engineer the risk label. In a real system, 
risk labels would come from independent expert assessment — 
making the prediction task harder and more meaningful.

---

## 🔬 Limitations and What Comes Next

This is a first-pass model. The honest limitations are:

- **Label quality** — risk labels were engineered from outcomes, 
not independently assessed. A flood that killed few people due 
to good evacuation might still be high risk — the model 
wouldn't know that.

- **Missing context** — population density, building stock, 
infrastructure quality, and land use all drive property risk 
but aren't in this dataset.

- **Static model** — risk patterns change as climate changes. 
A model trained on 1970-2021 data may not reflect future risk 
distributions.

Addressing these limitations — through multimodal data fusion, 
satellite observations, and uncertainty-aware estimation — is 
exactly what more advanced research in this space investigates.

---

## 🚀 Live Demo

> API deployed at: *(coming soon)*

---

## 👩‍💻 Author

**Suditi Sharma**
MSc Computer Science (Data Science) — University of Strathclyde
Former Motor Claims Handler — Aviva (FTSE 100 Insurer)
[LinkedIn](https://www.linkedin.com/in/suditi-sharma) | 
[GitHub](https://github.com/SuditiSharma)