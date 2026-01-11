# Distribution Shift & Silent Failure Detection for Deployed ML Models

## Problem Statement
Machine learning models deployed in real-world systems often fail silently due to changes in input data distribution. These failures may not immediately reflect in standard performance metrics, leading to unsafe or suboptimal decisions.

This project implements a monitoring framework to detect data distribution shift and identify silent model failure before severe performance degradation occurs.

---

## Key Contributions
- Designed a reference–deployment monitoring pipeline for ML systems
- Implemented statistical drift detection using Population Stability Index (PSI) and Kolmogorov–Smirnov tests
- Simulated real-world covariate shift in deployment data
- Demonstrated silent failure where data drift precedes visible performance collapse
- Built rule-based alerting logic for early failure detection

---

## System Overview
1. **Reference Distribution Modeling**
   - Training data used to learn baseline feature distributions
2. **Deployment Monitoring**
   - Incoming data processed in time-ordered batches
3. **Drift Detection**
   - Feature-wise PSI and KS tests computed per batch
4. **Performance Tracking**
   - Accuracy and recall tracked across batches
5. **Alerting Mechanism**
   - Alerts raised based on drift–performance interaction

---

## Dataset
- Diabetes Prediction Dataset (DPD)
- Large-scale, real-world tabular healthcare dataset

---

## Key Results
- Stable distributions show no false alerts
- Significant covariate shift detected via PSI (>2.5)
- Accuracy degrades after drift, confirming silent failure
- Alert system correctly identifies model failure onset

---

## Technologies Used
- Python
- NumPy, Pandas
- Scikit-learn
- SciPy
- Matplotlib

---

## How to Run
```bash
pip install -r requirements.txt
