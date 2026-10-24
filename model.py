"""Model loading, image preprocessing and prediction helpers."""
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image

# Loading model
model = load_model("digit_model.h5")


def preprocess_img(img_path):
    """Open an image, resize it to 224x224 and return a normalised array of shape (1, 224, 224, 3)."""
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape


def predict_result(predict):
    """Run the model on a preprocessed image and return the predicted digit."""
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)