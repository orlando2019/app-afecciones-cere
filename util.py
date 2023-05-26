import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np

def classify(image, model, class_names):
    """
    Esta función toma una imagen, un modelo y una lista de nombres de clases y devuelve la clase y la confianza 
	predichas.puntuación de la imagen.

    Parámetros:
    image (PIL.Image.Image): Una imagen para ser clasificada.
    model (tensorflow.keras.Model): un modelo de aprendizaje automático entrenado para la clasificación de 			 
	imágenes.

    class_names (lista): una lista de nombres de clases correspondientes a las clases que el modelo puede predecir.

    Devoluciones:
    Una tupla del nombre de clase pronosticado y la puntuación de confianza para esa predicción.
    """
    # convert image to (150, 150)
    image = ImageOps.fit(image, (150, 150), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # set model input
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    prediction = model.predict(data)
    # index = np.argmax(prediction)
    index = 0 if prediction[0][0] > 0.95 else 1
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score
