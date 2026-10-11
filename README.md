# 🏠 House Price Prediction

## About the Project

House Price Prediction is a Machine Learning project that predicts
median house values using numerical features from the California
Housing dataset.

The project demonstrates data exploration, preprocessing, model
training, evaluation, model comparison, prediction, and visualization.

## Dataset

- **Dataset:** California Housing
- **Source:** Scikit-learn
- **Total Records:** 20,640
- **Input Features:** 8
- **Target:** Median house value

### Input Features

- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Average Occupancy
- Latitude
- Longitude

The predictions represent median house values for California areas,
not the prices of individual properties.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Machine Learning Models

Three regression models were compared:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Random Forest achieved the best cross-validation R² score among
the three models.

## Model Performance

| Model | MAE | RMSE | Test R² | CV Mean R² |
|---|---:|---:|---:|---:|
| Linear Regression | 0.5332 | 0.7456 | 0.5758 | 0.6115 |
| Random Forest | 0.3265 | 0.5048 | 0.8055 | 0.8047 |
| Gradient Boosting | 0.3716 | 0.5422 | 0.7756 | 0.7865 |

The Random Forest model achieved a cross-validation R² score of
0.8047.

The compact Random Forest model uses restricted tree depth and
minimum leaf size to reduce model size.

**Note:** The target variable is measured in units of $100,000.
The model estimates area-level median house values and should not
be treated as a professional property valuation.

## Project Workflow

1. Load the California Housing dataset.
2. Explore and prepare the data.
3. Separate input features and the target variable.
4. Split the data into training and testing sets.
5. Train and compare three regression models.
6. Evaluate models using MAE, MSE, RMSE, and R².
7. Select the best-performing model using cross-validation.
8. Save the trained model.
9. Predict median house values using user input.
10. Visualize actual versus predicted values.
11. Display feature importance for the Random Forest model.

## Visualizations

The project generates the following visualizations:

- `actual_vs_predicted.png` — compares actual and predicted values.
- `feature_importance.png` — shows the relative importance of input features.

## Project Structure

```text
House Price Prediction/
├── data/
│   └── housing.csv
├── download_data.py
├── main.py
├── train.py
├── predict.py
├── visualize.py
├── compare.py
├── house_price_model.pkl
├── scaler.pkl
├── best_house_price_model.pkl
├── compact_house_price_model.pkl
├── model_comparison.csv
├── actual_vs_predicted.png
├── feature_importance.png
├── requirements.txt
└── README.md