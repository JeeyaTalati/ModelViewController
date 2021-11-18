from flask import Flask, jsonify, request
from Classifier import get_prediction

app = Flask(__name__)

@app.route("/")
def index():
    return "hello World"

@app.route("/predict-digit", methods = ["POST"])
def predict_digit():
    image = request.files.get("digit")
    prediction = get_prediction(image)
    return jsonify({
        "prediction": prediction,
        "status": "success"
    })



if __name__ == "__main__":
    app.run(debug = True)

