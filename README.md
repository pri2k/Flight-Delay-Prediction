# **Flight Delay Prediction using Ensemble Models**

## **Objectives**

* Predict whether a flight will be **Delayed** or **On-Time** using real-world operational flight data.
* Build a binary classification model using **Random Forest** and **XGBoost**.
* Apply preprocessing, cross-validation, hyperparameter tuning, and performance comparison.

---

## **Introduction**

Flight delays impact airline operations, customer satisfaction, and revenue.
Using machine learning, delays can be predicted in advance based on flight schedules, carrier information, and operational timing.
This project builds an ensemble-based prediction system for classifying flights as delayed or on-time.

---

## **Real World Dataset**

The dataset comes from the **MFDD (Multi-Modal Flight Delay Dataset)**, a large aviation dataset published in a NeurIPS research study.
Key columns include carrier, flight number, origin/destination airports, scheduled/actual departure time, taxi-out time, wheels-off time, and departure delay.

**Target Variable:**

```
Delayed = 1 if DEP_DELAY > 15 minutes  
Delayed = 0 otherwise
```

---

## **Data Preprocessing**

* Parsed timestamps and extracted features: month, weekday, scheduled hour, actual hour, wheels-off hour.
* Filled missing values (median for numeric, mode for categorical).
* Label-encoded categorical features (`OP_CARRIER`, `ORIGIN`, `DEST`).
* Exported a cleaned dataset for modeling.

---

## **Ensemble Models**

### **Random Forest**

* Bagging-based algorithm
* Strong baseline for tabular data

### **XGBoost**

* Gradient boosting
* Handles imbalance with `scale_pos_weight`
* Faster with GPU acceleration

---

## **Model Tuning and Evaluation**

* Used **5-fold Stratified Cross-Validation**.
* Hyperparameter tuning via **RandomizedSearchCV**.
* Evaluation metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC-AUC
  * Confusion Matrix
* Final comparison table saved as `model_comparison_metrics.csv`.

---

## **Deployment**

Trained models saved as:

* `best_random_forest.pkl`
* `best_xgboost.pkl`

They can be deployed using:

* Flask / FastAPI APIs
* Streamlit apps
* Cloud services like AWS/GCP/Azure

---

## **Final Report Conclusion**

This project shows that ensemble models—especially **XGBoost**—perform effectively for flight delay prediction.
With proper preprocessing, tuning, and evaluation, the classifier achieves strong predictive accuracy and is suitable for real-world aviation analytics and decision-making.

