from cv2 import cv2
import tensorflow as tf
from tensorflow.keras.applications import densenet
from src.diagnosis_model.classification import DiagnosisModel
from src.diagnosis_model.configs import IMAGE_CLS_SIZE


def load_model(ckpt: str = None):
    model = DiagnosisModel()
    if ckpt is not None:
        model.load_weights(ckpt).expect_partial()
    return model


def predict(model: DiagnosisModel, filename):
    image = cv2.imread(filename)
    if image is None:
        return None
    image = cv2.resize(image, dsize=(IMAGE_CLS_SIZE, IMAGE_CLS_SIZE))
    image = tf.convert_to_tensor(image, tf.float32)
    image = tf.expand_dims(image, axis=0)
    image = densenet.preprocess_input(image)
    results = model(image, training=False)
    return tf.nn.sigmoid(results).numpy()
