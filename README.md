# Airbnb Destination Prediction 


[![Python Version](https://img.shields.io/badge/python-3.7+-brightgreen.svg)](https://www.python.org/downloads/release/python-379/)
[![NumPy Version](https://img.shields.io/badge/NumPy-1.19.2-red.svg)](https://numpy.org/doc/stable/reference/generated/numpy.version.html)
[![Pandas Version](https://img.shields.io/badge/pandas-1.2.4-red.svg)](https://pandas.pydata.org/pandas-docs/stable/whatsnew/v1.2.4.html)
[![Scikit-learn Version](https://img.shields.io/badge/scikit--learn-0.24.1-blue.svg)](https://scikit-learn.org/stable/whats_new/v0.24.html)
[![Seaborn Version](https://img.shields.io/badge/seaborn-0.11.1-green.svg)](https://seaborn.pydata.org/docs/0.11.1/whatsnew.html)
[![Streamlit Version](https://img.shields.io/badge/streamlit-0.73.1-green.svg)](https://github.com/streamlit/streamlit/releases/tag/v0.73.1)
[![Git Version](https://img.shields.io/badge/Git-2.28.0-red.svg)](https://github.com/git/git/commit/2.28.0)


An end-to-end machine learning pipeline for predicting a user's first booking destination using demographic attributes and raw clickstream session behavior. This project addresses a cold-start personalization problem in the travel domain, where predictions are made using limited early-stage user information to improve destination recommendation and booking conversion.

---

# Project Overview

A global travel platform aims to predict where a newly registered user is most likely to make their first booking.

The challenge is to infer destination preference before the user completes extensive browsing or booking actions. This enables early personalization of homepage recommendations, targeted marketing campaigns, and improved conversion rates.

The project uses:

- User demographic information
- Signup metadata
- Raw clickstream session logs

The target is a multi-class classification problem with 12 destination classes, including:

- NDF (No Destination Found)
- US
- Other international destinations

This dataset presents major challenges such as:

- Extreme class imbalance
- Large-scale clickstream aggregation
- Cold-start users with no session history
- Text-heavy categorical behavioral signals

---

# Business Objective

The objective is to build a machine learning system that predicts a user's likely first booking destination immediately after signup.

## Business impact:

- Improve homepage personalization
- Increase booking conversion rate
- Enable targeted destination-based marketing
- Reduce generic recommendation exposure

The business insights we discovered during research are noted in the Report whose link is attached below:- 

###  [Business Report ](./docs/The Digital%20Footprint%20User%20Journey.pdf)
---

# Problem Statement

Convert raw user-level and session-level behavioral data into predictive user representations for destination prediction.

The system must:

- Handle users with session history
- Handle users without session history
- Extract signal from clickstream actions
- Work under severe class imbalance

---


# Steps to Run :- 

### Running with Docker Compose (Recommended)

#### Build and start all services

```bash
cd Docker-Prod-Model/airbnb-predictor
docker-compose up --build
```
#### Stop

```bash
docker-compose down
```
#### Stop and remove volumes
```bash
docker-compose down -v
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend  | http://localhost:4000 |
| ML API   | http://localhost:8000 |
| MongoDB  | mongodb://localhost:27017 |

---

##  Running Locally (Without Docker)

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB running locally on port 27017

### 1. ML Service

```bash
cd Docker-Prod-Model/airbnb-predictor
cd ml-service
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Verify: http://localhost:8000/health

Docs:   http://localhost:8000/docs

### 2. Backend

```bash
cd ..
cd backend
cp .env.example .env        # edit if needed
npm install
npm run dev                 # or: npm start
```

Service runs at:-  http://localhost:4000

### 3. Frontend

```bash
cd ..
cd frontend
cp .env.example .env        # set REACT_APP_BACKEND_URL if needed
npm install
npm start
```

App runs at http://localhost:3000
(CRA proxy forwards /api/* to backend automatically in dev)

---

# Workflow

## Sprint 1: Discovery & Baseline

- Understand business problem
- Profile datasets
- Clean user data
- Aggregate sessions
- Build baseline model

## Sprint 2: Refinement & Modeling

- Advanced feature engineering
- Text vectorization
- Model comparison
- Error analysis
- Demo preparation

---

# Feature Engineering Strategy

## User Features

- Age buckets
- Signup method encoding
- Affiliate source grouping
- Browser categories

## Session Aggregates

- Total actions
- Unique actions
- Total seconds elapsed
- Average session time
- First action
- Last action
- Device diversity

## Behavioral Ratios

- Search ratio
- Booking related ratio
- Messaging ratio

## Text Features

User action sequence treated as document:

Example:

search_results click listing message

TF-IDF / CountVectorizer applied on action sequences.

---

# Modeling Approach

## Baseline Model

- Logistic Regression

## Compared Models

- Random Forest
- XGBoost

# Critical Challenges

## Class Imbalance

- NDF dominates target distribution
- Minority destinations difficult to learn

## Cold Start Problem

Some users have no session data.

## Session Aggregation

Multiple rows per user must be converted to one predictive vector.

## Sparse Text Signals

Action labels generate high dimensional sparse features.

---

# Design Documents

## [HLD (High Level Design)](./docs/HLD.md)

![System High-Level Design](./docs/assets/HLD.png)


Contains:

- business architecture
- data flow
- prediction pipeline
- assumptions

## [LLD (Low Level Design)](./docs/LLD.md)
Contains:

- feature inventory
- preprocessing plan
- model hypotheses
- validation logic

---

# Future Improvements

- Sequence modeling using action order
- Recency-weighted session features
- Rare-class boosting
- Two-model architecture for no-session users

---

# Contributors

- [Niranjan Hebli](https://github.com/NiranjanHebli)

- [Sudhanshu Biswas](https://github.com/SudhanshuBiswas01)

- [Het Patel](https://github.com/Het0808)

- [Anirudh Sharma](https://github.com/anirudhkrishnatreya)

- [Avik Chatterjee](https://github.com/Avichatt)