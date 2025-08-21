from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model
MODEL_PATH = "weather_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Expected features
FEATURES = ["precipitation", "temp_max", "temp_min", "wind"]

@app.route("/")
def home():
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form data
        data = [float(request.form[feat]) for feat in FEATURES]
        df = pd.DataFrame([data], columns=FEATURES)

        pred = model.predict(df)[0]
        return render_template("index.html", features=FEATURES, result=f"Prediction: {pred}")
    except Exception as e:
        return render_template("index.html", features=FEATURES, result=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
