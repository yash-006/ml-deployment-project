import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib


# --------------------------------------------------
# 1. Create a small dataset
# --------------------------------------------------

data = {
    "age": [
        22, 25, 28, 30, 32,
        35, 38, 40, 42, 45,
        48, 50, 52, 55, 58,
        23, 27, 31, 36, 44
    ],
    "salary": [
        25000, 30000, 35000, 40000, 45000,
        50000, 55000, 60000, 65000, 70000,
        75000, 80000, 85000, 90000, 95000,
        28000, 38000, 42000, 58000, 68000
    ],
    "approved": [
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        0, 0, 0, 1, 1
    ]
}

df = pd.DataFrame(data)


# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

X = df[["age", "salary"]]
y = df["approved"]


# --------------------------------------------------
# 3. Split data into training and testing sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 4. Create and train the model
# --------------------------------------------------

model = LogisticRegression()

model.fit(X_train, y_train)


# --------------------------------------------------
# 5. Evaluate the model
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")


# --------------------------------------------------
# 6. Save the trained model
# --------------------------------------------------

joblib.dump(model, "model/approval_model.joblib")

print("Model saved successfully!")