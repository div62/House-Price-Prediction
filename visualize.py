import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/housing.csv")

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load the best model
model = joblib.load("best_house_price_model.pkl")

# Predict house values
y_pred = model.predict(X_test)

# Actual vs Predicted chart
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.4)

plt.xlabel("Actual Median House Value ($100,000 units)")
plt.ylabel("Predicted Median House Value ($100,000 units)")
plt.title("Random Forest: Actual vs Predicted House Values")

plt.plot([0, 5], [0, 5], "r--")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

print("Actual vs Predicted chart saved successfully!")

# Feature Importance chart
importance = model[-1].feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance", ascending=True
)

plt.figure(figsize=(8, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("Feature Importance chart saved successfully!")