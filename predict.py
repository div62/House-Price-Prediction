import joblib
import pandas as pd
model = joblib.load("compact_house_price_model.pkl")
print("\n===== House Price Prediction =====")

try:
    med_inc = float(input("Enter median income: "))
    house_age = float(input("Enter house age: "))
    ave_rooms = float(input("Enter average rooms: "))
    ave_bedrms = float(input("Enter average bedrooms: "))
    population = float(input("Enter population: "))
    ave_occup = float(input("Enter average occupancy: "))
    latitude = float(input("Enter latitude: "))
    longitude = float(input("Enter longitude: "))

    # Prepare input data
    data = pd.DataFrame([[
        med_inc,
        house_age,
        ave_rooms,
        ave_bedrms,
        population,
        ave_occup,
        latitude,
        longitude
    ]], columns=[
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude"
    ])

    # Predict house value
    prediction = model.predict(data)[0]

    # Convert prediction into dollars
    price = prediction * 100000

    print(f"\nPredicted median house value: ${price:,.2f}")

except ValueError:
    print("\nInvalid input! Please enter numeric values.")
