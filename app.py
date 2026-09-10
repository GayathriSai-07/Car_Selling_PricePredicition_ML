from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("models/car_price_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    prediction = None
    error = None

    # Stores the values entered by the user
    # so that they remain inside the input boxes.
    form_data = {
        "present_price": "",
        "kms_driven": "",
        "car_age": "",
        "fuel_type": "",
        "seller_type": "",
        "transmission": "",
        "owner": ""
    }

    if request.method == "POST":

        # Get submitted values
        form_data = {
            "present_price": request.form.get("present_price", ""),
            "kms_driven": request.form.get("kms_driven", ""),
            "car_age": request.form.get("car_age", ""),
            "fuel_type": request.form.get("fuel_type", ""),
            "seller_type": request.form.get("seller_type", ""),
            "transmission": request.form.get("transmission", ""),
            "owner": request.form.get("owner", "")
        }

        try:

            # Convert numeric values
            present_price = float(form_data["present_price"])
            kms_driven = float(form_data["kms_driven"])
            car_age = int(form_data["car_age"])
            owner = int(form_data["owner"])

            fuel_type = form_data["fuel_type"]
            seller_type = form_data["seller_type"]
            transmission = form_data["transmission"]

            # -------------------------
            # VALIDATION
            # -------------------------

            if present_price <= 0:
                raise ValueError(
                    "Present Price must be greater than 0."
                )

            if kms_driven < 0:
                raise ValueError(
                    "Kilometers Driven cannot be negative."
                )

            if car_age < 0:
                raise ValueError(
                    "Car Age cannot be negative."
                )

            if owner < 0 or owner > 3:
                raise ValueError(
                    "Previous Owners must be between 0 and 3."
                )

            if fuel_type not in ["Petrol", "Diesel", "CNG"]:
                raise ValueError(
                    "Please select a Fuel Type."
                )

            if seller_type not in ["Dealer", "Individual"]:
                raise ValueError(
                    "Please select a Seller Type."
                )

            if transmission not in ["Manual", "Automatic"]:
                raise ValueError(
                    "Please select a Transmission."
                )

            # -------------------------
            # DATA FOR MODEL
            # -------------------------

            input_data = pd.DataFrame({
                "Present_Price": [present_price],
                "Kms_Driven": [kms_driven],
                "Car_Age": [car_age],
                "Fuel_Type": [fuel_type],
                "Seller_Type": [seller_type],
                "Transmission": [transmission],
                "Owner": [owner]
            })

            # -------------------------
            # PREDICTION
            # -------------------------

            prediction = model.predict(input_data)[0]

            prediction = round(float(prediction), 2)

        except ValueError as e:

            error = str(e)

        except Exception as e:

            print("Prediction Error:", e)

            error = (
                "Prediction could not be completed. "
                "Please check your details."
            )

    return render_template(
        "predict.html",
        prediction=prediction,
        error=error,
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True)