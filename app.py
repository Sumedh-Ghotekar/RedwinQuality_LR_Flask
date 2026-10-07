import joblib
import pandas as pd

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model
obj = joblib.load("redwine.joblib")
model = obj["model"]
columns = obj["columns"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET"])
def predict():

    try:
        data = {
            "fixed acidity": [float(request.args.get("fixed_acidity"))],
            "volatile acidity": [float(request.args.get("volatile_acidity"))],
            "citric acid": [float(request.args.get("citric_acid"))],
            "residual sugar": [float(request.args.get("residual_sugar"))],
            "chlorides": [float(request.args.get("chlorides"))],
            "free sulfur dioxide": [float(request.args.get("free_sulfur_dioxide"))],
            "total sulfur dioxide": [float(request.args.get("total_sulfur_dioxide"))],
            "density": [float(request.args.get("density"))],
            "pH": [float(request.args.get("pH"))],
            "sulphates": [float(request.args.get("sulphates"))],
            "alcohol": [float(request.args.get("alcohol"))]
        }

        df = pd.DataFrame(data)

        # Arrange columns same as training
        df = df[columns]

        prediction = model.predict(df)

        return jsonify({
            "prediction": float(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )