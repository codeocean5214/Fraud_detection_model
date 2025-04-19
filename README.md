# 🛡️ Credit Card Fraud Detection Using Logistic Regression

This project applies **Logistic Regression** to detect fraudulent credit card transactions in a highly imbalanced dataset of 80,000 records. A custom evaluation metric is used to ensure balance between **precision** and **recall**, targeting high model effectiveness in fraud detection.

---

## 📌 Problem Statement

Credit card fraud detection presents a major challenge due to class imbalance. A highly accurate model might still miss fraudulent cases. Hence, rather than relying solely on accuracy, this project introduces a custom evaluation function that focuses on the **minimum of precision and recall** to ensure both are sufficiently high.

---

## 🚀 Technologies Used

- Python 3.13
- NumPy
- pandas
- scikit-learn
- matplotlib

---

## 🧠 Model and Scoring Strategy

We use **Logistic Regression** with `GridSearchCV` to fine-tune the `class_weight` hyperparameter. The custom evaluation function `min_both` ensures that both precision and recall are optimized equally:

```python
def min_both(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return min(recall, precision)
```

### ⚙️ Grid Search Setup

```python
grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid={'class_weight': [{0: 1, 1: v} for v in np.linspace(0, 20, 30)]},
    scoring={
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'both_min': make_scorer(min_both)
    },
    refit='both_min',
    return_train_score=True,
    cv=5,
    n_jobs=-1
)

grid.fit(X, y)
```

---

## 📊 Visualization

We visualize model performance across different class weights using:

```python
plt.figure(figsize=(12, 4))
for score in ['mean_test_recall', 'mean_test_precision', 'mean_train_both_min']:
    plt.plot([_[1] for _ in df['param_class_weight']], df[score], label=score)
plt.legend()
plt.title("Performance Metrics vs Class Weight for Fraud Detection")
plt.xlabel("Class Weight for Fraudulent Class")
plt.ylabel("Score")
plt.grid(True)
```

---

## ✅ Results Summary

- Logistic Regression model was fine-tuned for optimal performance on imbalanced data.
- The use of `min(precision, recall)` ensured balanced optimization.
- Achieved interpretable metrics with visual insight into the effect of class weighting.

---

## 📁 Repository Structure

```
├── credit_fraud_model.py      # Main model training script
├── min_both.py                # Custom scoring function
├── plots/                     # Graphs and analysis visuals
├── README.md                  # Project documentation
```

---

## 📌 Future Enhancements

- Try ensemble models like Random Forest and XGBoost
- Implement anomaly detection techniques
- Explore oversampling methods like SMOTE or ADASYN
- Deploy model as an API for real-time predictions

---

## 🙌 Author

**Aditya Mishra**  
📧 [Email](mailto:workingmishraji@gmail.com)  
🔗 [GitHub](https://github.com/yourusername) | [LinkedIn](https://www.linkedin.com/in/yourprofile)

---

> If you found this useful, feel free to ⭐ the repo and share it!
