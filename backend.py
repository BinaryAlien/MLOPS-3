from flask import Flask, request
from http import HTTPStatus
import mlflow
import numpy as np
import os


MLFLOW_HOST = os.getenv("MLFLOW_HOST", "127.0.0.1")
MLFLOW_PORT = os.getenv("MLFLOW_PORT", "8080")


def load_model(model_uri):
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Load model {model.metadata.model_uuid}")
    return model


mlflow.set_tracking_uri(uri=f"http://{MLFLOW_HOST}:{MLFLOW_PORT}")
model = load_model("runs:/80b9fd3513d245f6b1ffb9387d39ca7f/iris_model")


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
    outputs = model.predict(inputs)
    return {"class": int(outputs[0])}


@app.route("/update-model", methods=["POST"])
def update_model():
    global model
    model_uri = str(request.json["model_uri"])
    model = load_model(model_uri)
    return model.metadata.to_dict()
