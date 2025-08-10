import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv(r"C:\Users\Betraaj\Downloads\Desktop\elevate labs\TASK 4\data.csv")

# Drop ID and unnamed column
df = df.drop(columns=['id', 'Unnamed: 32'])

# Encode target: M = 1, B = 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# 2. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Fit Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Predictions
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred_prob >= 0.5).astype(int)

# 6. Evaluation metrics
cm = confusion_matrix(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print("\n=== Model Evaluation ===")
print("Confusion Matrix:")
print(cm)
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")


# 7. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 8. Threshold tuning example
new_threshold = 0.4
y_pred_new = (y_pred_prob >= new_threshold).astype(int)
print(f"Precision @ {new_threshold}: {precision_score(y_test, y_pred_new):.3f}")
print(f"Recall @ {new_threshold}: {recall_score(y_test, y_pred_new):.3f}")

"""
EXPLANATION:
- Logistic regression predicts probability using the sigmoid function:
    σ(z) = 1 / (1 + e^(-z))
- Threshold decides classification (default = 0.5).
- Precision: correct positives / all predicted positives.
- Recall: correct positives / all actual positives.
- ROC-AUC: measures ability to distinguish between classes.
"""
