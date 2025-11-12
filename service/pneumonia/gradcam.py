# pneumonia_service/gradcam.py
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model

class GradCAM:
    def __init__(self, base_model, last_conv_layer_name, tamanho_img=(224,224)):
        self.base_model = base_model
        self.tamanho_img = tamanho_img

        # encontra última conv
        self.last_conv = None
        for layer in reversed(self.base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) and layer.name == last_conv_layer_name:
                self.last_conv = layer
                break
        if not self.last_conv:
            raise ValueError("Última camada conv não encontrada!")

    def calcular_cam(self, img_array, n_samples=30, noise_level=0.2, power=3):
        grad_model = Model(
            inputs=self.base_model.input,
            outputs=[self.last_conv.output, self.base_model.output]
        )

        cam_accum = None
        for _ in range(n_samples):
            noisy_input = img_array.copy()
            if n_samples > 1:
                noise = np.random.normal(0, noise_level * 25, img_array.shape)
                noisy_input = img_array + noise.astype(np.float32)

            with tf.GradientTape() as tape:
                conv_out, feature_maps = grad_model(noisy_input, training=False)
                pred = tf.reduce_mean(feature_maps, axis=(1, 2))
                class_score = tf.maximum(pred, 1 - pred)

            grads = tape.gradient(class_score, conv_out)
            if grads is None:
                continue

            weights = tf.reduce_mean(grads, axis=(1, 2))
            cam = tf.reduce_sum(weights * conv_out[0], axis=-1)

            if cam_accum is None:
                cam_accum = cam
            else:
                cam_accum += cam

        cam = cam_accum / n_samples
        cam = np.maximum(cam, 0)
        cam = cam**power
        cam /= (cam.max() + 1e-8)

        h, w = self.tamanho_img
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        cam = cv2.GaussianBlur(cam, (5,5), sigmaX=2)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    @staticmethod
    def overlay(img_bgr, heatmap, alpha=0.7):
        heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]), cv2.INTER_CUBIC)
        # vermelho mais intenso no pico
        heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_HOT).astype(np.float32)
        heatmap_color[:,:,2] *= heatmap
        heatmap_color[:,:,0] *= 0.2
        heatmap_color[:,:,1] *= 0.4
        heatmap_color = np.clip(heatmap_color,0,255).astype(np.uint8)
        return cv2.addWeighted(img_bgr, 1-alpha, heatmap_color, alpha, 0)
