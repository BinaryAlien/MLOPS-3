#!/usr/bin/env python3

import os
import requests
import unittest


BACKEND_HOST = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
BACKEND_PORT = os.getenv("FLASK_RUN_PORT", "5000")
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"


CLASS_SETOSA = 0
CLASS_VERSICOLOR = 1
CLASS_VIRGINICA = 2


def predict(**sample):
    response = requests.post(BACKEND_URL + "/predict", json=sample)
    response.raise_for_status()
    output = response.json()
    return int(output["class"])


def update_model(model_uri):
    response = requests.post(
        BACKEND_URL + "/update-model", json={"model_uri": model_uri}
    )
    response.raise_for_status()
    metadata = response.json()
    return metadata


class TestPredict(unittest.TestCase):
    def _test_predict(self, label, **sample):
        response = requests.post(BACKEND_URL + "/predict", json=sample)
        response.raise_for_status()
        output = response.json()
        self.assertEqual(label, output["class"])

    def test_setosa(self):
        self.assertEqual(
            CLASS_SETOSA,
            predict(
                sepal_length=5.7,
                sepal_width=3.8,
                petal_length=1.7,
                petal_width=0.3,
            ),
        )

    def test_versicolor(self):
        self.assertEqual(
            CLASS_VERSICOLOR,
            predict(
                sepal_length=6.1,
                sepal_width=2.8,
                petal_length=4.7,
                petal_width=1.2,
            ),
        )

    def test_virginica(self):
        self.assertEqual(
            CLASS_VIRGINICA,
            predict(
                sepal_length=7.7,
                sepal_width=2.6,
                petal_length=6.9,
                petal_width=2.3,
            ),
        )


class TestUpdateModel(unittest.TestCase):
    def test_update_v1(self):
        metadata = update_model("runs:/5059a130c117484e9f2f1049b8863aa5/iris_model")
        self.assertEqual(
            "b58cb3b5633945608ad7bb67cc41bd5c",
            metadata["next"]["model_uuid"],
        )

    def test_update_v2(self):
        metadata = update_model("runs:/80b9fd3513d245f6b1ffb9387d39ca7f/iris_model")
        self.assertEqual(
            "5d48e96a3ce84f898b67902b534b8c6c",
            metadata["next"]["model_uuid"],
        )


if __name__ == "__main__":
    unittest.main()
