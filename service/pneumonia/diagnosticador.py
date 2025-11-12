# pneumonia_service/diagnosticador.py
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

class Diagnosticador:
    def __init__(self, caminho_modelo, tamanho_img=(224,224)):
        self.tamanho_img = tamanho_img
        self.model = load_model(caminho_modelo)

        dummy = np.zeros((1,*tamanho_img,3), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)

    def preprocessar(self, caminho_imagem):
        img = image.load_img(caminho_imagem, target_size=self.tamanho_img)
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = tf.keras.applications.resnet50.preprocess_input(arr.copy())
        return img, arr

    def diagnosticar(self, arr):
        pred = float(self.model.predict(arr, verbose=0)[0][0])
        classe = "PNEUMONIA" if pred > 0.5 else "NORMAL"
        confianca = pred if pred > 0.5 else 1 - pred
        return classe, confianca
