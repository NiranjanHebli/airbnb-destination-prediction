# System High-Level Design (HLD)

## Overview
This document describes the high-level architecture of a machine learning system that handles user data, performs preprocessing, trains models, and serves predictions through an inference pipeline.

The system supports:
- User registration and data collection
- Data preprocessing and session aggregation
- Handling class imbalance
- Model training, evaluation, and registry
- Prediction serving through an inference pipeline
- Experiment tracking and monitoring

---
![System High-Level Design](./assets/HLD.png)

## Architecture Components

### 1. End User
The End User interacts with the system by:
- Registering into the platform
- Receiving predictions via the UI

---

### 2. User Database
Uses MongoDB to store:
- User registration data
- Raw user information

Role:
- Acts as the primary persistent storage for user data

---

### 3. Registration Logic
Handles:
- User onboarding
- Writing user data into the database

Flow:
End User -> Registration Logic -> MongoDB

---

### 4. Data Source Logic
Responsible for:
- Fetching raw user data from MongoDB
- Preparing it for preprocessing

---

### 5. Data Preprocessing and Session Aggregation
Performs:
- Data cleaning
- Feature extraction
- Session-level aggregation

Output:
- Preprocessed dataset ready for training

---

### 6. Class Imbalance Handling

Decision: Class Imbalance?

If YES:
- Applies SMOTE and class balancing logic
- Produces a balanced dataset

If NO:
- Proceeds directly to model training

---

### 7. Model Training
Models used:
- Random Forest
- XGBoost

Input:
- Preprocessed or balanced dataset

Output:
- Trained models

---

### 8. Experiment Tracker
Tracks:
- Hyperparameters
- Training metrics
- Experiment runs

Examples:
- MLflow
- Weights & Biases

---

### 9. Model Registry (Staging / Production)
Stores:
- Trained models
- Model versions

Responsibilities:
- Register new models
- Manage staging and production models

---

### 10. Model Evaluation
Evaluates models using:
- F1 Score
- Confusion Matrix

Purpose:
- Select the best-performing model (champion)

---

### 11. Inference Service (ML Pipeline)
Handles:
- Loading the best model from the registry
- Running inference on incoming data

---

### 12. Prediction UI
Displays:
- Model predictions to the end user

---

### 13. ML Engineer
Responsible for:
- Monitoring experiments
- Tracking performance via the experiment tracker

---

## End-to-End Flow

### Training Pipeline

1. User registers and data is stored in MongoDB  
2. Data Source Logic fetches raw data  
3. Data is preprocessed and aggregated  
4. Class imbalance is checked  
5. If required, SMOTE is applied  
6. Model is trained  
7. Metrics are logged in the experiment tracker  
8. Model is registered in the model registry  
9. Model is evaluated and the best model is selected  

---

### Inference Pipeline

1. Best model is loaded from the model registry  
2. Input data is passed to the inference service  
3. Predictions are generated  
4. Results are shown in the Prediction UI  
5. User receives predictions  

---

## Monitoring and Feedback

- ML Engineer monitors:
  - Model performance
  - Training metrics
- Enables continuous improvement of models

---

## Key Design Decisions

- Separation of training and inference pipelines
- Use of model registry for version control
- Explicit handling of class imbalance
- Experiment tracking for reproducibility
- Use of MongoDB for scalable NoSQL storage

---

## Future Improvements

- Introduce a feature store (online and offline)
- Add real time streaming (e.g., Kafka)
- Implement model and data drift monitoring
- Enable automated retraining pipelines

---
