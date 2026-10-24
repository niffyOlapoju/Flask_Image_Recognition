"""Automated tests for the hand sign digit Flask application."""
import io
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from app import app as flask_app
from model import preprocess_img, predict_result


def make_image_bytes(size=(50, 50), color=(255, 0, 0)):
    """Create a small in-memory PNG image for upload tests."""
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


@pytest.fixture(name="client")
def client_fixture():
    """Return a Flask test client."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as test_client:
        yield test_client


def test_home_page_loads(client):
    """The home page can be accessed successfully."""
    response = client.get("/")
    assert response.status_code == 200


def test_valid_upload_returns_prediction(client):
    """A valid image upload returns a result page showing the prediction."""
    with patch("app.predict_result", return_value="MOCK-PRED-42"):
        response = client.post(
            "/prediction",
            data={"file": (make_image_bytes(), "digit.png")},
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    assert b"MOCK-PRED-42" in response.data


def test_missing_file_is_handled(client):
    """A request with no uploaded file shows an error instead of crashing."""
    response = client.post("/prediction", data={})
    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_invalid_file_type_is_handled(client):
    """A text file uploaded as an image shows an error instead of crashing."""
    response = client.post(
        "/prediction",
        data={"file": (io.BytesIO(b"this is not an image"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_empty_file_is_handled(client):
    """An empty file with an image name shows an error instead of crashing."""
    response = client.post(
        "/prediction",
        data={"file": (io.BytesIO(b""), "empty.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_prediction_route_rejects_get(client):
    """The prediction route only accepts POST requests."""
    response = client.get("/prediction")
    assert response.status_code == 405


def test_preprocess_output_shape():
    """Preprocessing resizes any image to a single 224x224 RGB sample."""
    result = preprocess_img(make_image_bytes(size=(10, 10)))
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_values_are_normalised():
    """Preprocessed pixel values are scaled into the 0 to 1 range."""
    result = preprocess_img(make_image_bytes(color=(255, 255, 255)))
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_predict_result_returns_highest_probability_class():
    """The prediction function returns the index of the largest probability."""
    fake_output = np.array([[0.05, 0.1, 0.7, 0.05, 0.1]])
    with patch("model.model") as mock_model:
        mock_model.predict.return_value = fake_output
        digit = predict_result(np.zeros((1, 224, 224, 3)))
    assert digit == 2
