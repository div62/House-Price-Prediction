import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# Load dataset
data = pd.read_csv("data/housing.csv")

X = data.drop("MedHouseVal", axis=1)
y = data["MedHouseVal"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create model pipelines
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Random Forest": Pipeline([
    ("model", RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
]),
    
    "Gradient Boosting": Pipeline([
        ("model", GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        ))
    ])
}


# Compare models
results = []

print("\nMODEL COMPARISON RESULTS")
print("=" * 90)

for name, model in models.items():

    # Cross-validation on training data
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="r2"
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate on test data
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Test R2": r2,
        "CV Mean R2": cv_scores.mean(),
        "CV Std": cv_scores.std()
    })


# Display results
results_df = pd.DataFrame(results)

print(results_df.round(4).to_string(index=False))


# Select the best model using cross-validation
best_model_name = results_df.loc[
    results_df["CV Mean R2"].idxmax(), "Model"
]

best_model = models[best_model_name]


# Train the selected model on all training data
best_model.fit(X_train, y_train)
joblib.dump(best_model, "compact_house_price_model.pkl", compress=3)

# Save comparison results
results_df.to_csv("model_comparison.csv", index=False)


print("\n" + "=" * 90)
print("BEST MODEL:", best_model_name)
print("Cross-validation R2:", round(
    results_df.loc[
        results_df["Model"] == best_model_name, "CV Mean R2"
    ].iloc[0], 4
))

print("\nBest model saved as: best_house_price_model.pkl")
print("Comparison results saved as: model_comparison.csv")