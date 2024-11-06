from flask import Flask, request
from http import HTTPStatus
import mlflow
import numpy as np
import os


MLFLOW_HOST = os.getenv("MLFLOW_HOST", "127.0.0.1")
MLFLOW_PORT = os.getenv("MLFLOW_PORT", "8080")

PROBABILITY_USE_CURRENT = 0.75
PROBABILITY_USE_NEXT = 1.0 - PROBABILITY_USE_CURRENT


def load_model(model_uri):
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Load model {model.metadata.model_uuid}")
    return model


mlflow.set_tracking_uri(uri=f"http://{MLFLOW_HOST}:{MLFLOW_PORT}")

model_current = load_model("runs:/80b9fd3513d245f6b1ffb9387d39ca7f/iris_model")
model_next = model_current


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    sample = [
        float(request.json["sepal_length"]),
        float(request.json["sepal_width"]),
        float(request.json["petal_length"]),
        float(request.json["petal_width"]),
    ]
    inputs = np.array([sample])
    if np.random.random() < PROBABILITY_USE_NEXT:
        model = model_next
    else:
        model = model_current
    outputs = model.predict(inputs)
    return {"class": int(outputs[0])}


@app.route("/update-model", methods=["POST"])
def update_model():
    global model_next
    model_uri = str(request.json["model_uri"])
    model_next = load_model(model_uri)
    return {
        "current": model_current.metadata.to_dict(),
        "next": model_next.metadata.to_dict(),
    }


@app.route("/accept-next-model", methods=["POST"])
def accept_next_model():
    global model_current, model_next
    model_current = model_next
    return {
        "current": model_current.metadata.to_dict(),
        "next": model_next.metadata.to_dict(),
    }
