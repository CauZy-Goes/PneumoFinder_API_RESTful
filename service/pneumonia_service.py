# service/pneumonia_service.py
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


class DetectorDePneumoniaService:
    def __init__(self, caminho_modelo, tamanho_img=(224, 224)):
        self.tamanho_img = tamanho_img
        self.model = load_model(caminho_modelo)

        # FORÇA INICIALIZAÇÃO
        dummy = tf.zeros((1, *tamanho_img, 3))
        _ = self.model.predict(dummy, verbose=0)

        # === ENCONTRA BASE RESNET ===
        self.base_model = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model) and 'resnet' in layer.name.lower():
                self.base_model = layer
                break
        if not self.base_model:
            raise ValueError("ResNet50 não encontrada!")

        # === ÚLTIMA CONV ===
        self.last_conv = None
        for layer in reversed(self.base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                self.last_conv = layer
                break
        if not self.last_conv:
            raise ValueError("Conv2D não encontrada!")

        print(f"Grad-CAM target: {self.last_conv.name}")

    def _preprocessar(self, caminho_imagem):
        img = image.load_img(caminho_imagem, target_size=self.tamanho_img)
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = tf.keras.applications.resnet50.preprocess_input(arr.copy())
        return img, arr

    def _gradcam_smooth(self, img_array, n_samples=30, noise_level=0.2):
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

        # Destacar apenas o ponto central mais relevante
        cam = cam**2
        cam /= (cam.max() + 1e-8)

        h, w = self.tamanho_img
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        cam = cv2.GaussianBlur(cam, (3, 3), sigmaX=1.5)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def _apply_heatmap_red(self, cam):
        cam_uint8 = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_HOT)
        heatmap = heatmap.astype(np.float32)
        heatmap[:, :, 2] = np.clip(heatmap[:, :, 2] * cam, 0, 255)  # vermelho central
        heatmap[:, :, 0] *= 0.2
        heatmap[:, :, 1] *= 0.4
        return heatmap.astype(np.uint8)

    def _overlay(self, img_bgr, heatmap, alpha=0.68):
        heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]), cv2.INTER_CUBIC)
        return cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)

    def diagnosticar_com_explicacao(self, caminho_imagem, pasta_saida="relatorios"):
        nome = os.path.splitext(os.path.basename(caminho_imagem))[0]
        pasta = os.path.join(pasta_saida, nome)
        os.makedirs(pasta, exist_ok=True)

        img_pil, arr = self._preprocessar(caminho_imagem)
        pred = float(self.model.predict(arr, verbose=0)[0][0])
        classe = "PNEUMONIA" if pred > 0.5 else "NORMAL"
        confianca = pred if pred > 0.5 else 1 - pred

        print(f"{nome}: {classe} ({confianca:.1%})")

        cam = self._gradcam_smooth(arr)
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        heatmap = self._apply_heatmap_red(cam)
        overlay = self._overlay(img_bgr, heatmap, alpha=0.7)

        def salvar(fig, path):
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
            plt.close(fig)

        # 01 Original
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(img_pil)
        ax.set_title(f"Original\n{classe} ({confianca:.1%})", fontsize=14, pad=15, weight='bold')
        ax.axis('off')
        salvar(fig, f"{pasta}/01_original.png")

        # 02 Grad-CAM
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(cam, cmap='hot', vmin=0, vmax=1)
        ax.set_title("Grad-CAM (Foco Pulmonar)", fontsize=14, pad=15, weight='bold')
        ax.axis('off')
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("Ativação", rotation=270, labelpad=15)
        salvar(fig, f"{pasta}/02_gradcam.png")

        # 03 Overlay
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title("Regiões de Pneumonia", fontsize=14, pad=15, weight='bold')
        ax.axis('off')
        salvar(fig, f"{pasta}/03_overlay.png")

        # 04 Confiança
        fig, ax = plt.subplots(figsize=(5,4))
        bars = ax.bar(['NORMAL', 'PNEUMONIA'], [1-pred, pred], color=['#2ca02c', '#d62728'], edgecolor='black')
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.03, f'{h:.1%}', ha='center', fontweight='bold', fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title("Confiança", fontsize=14, weight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        salvar(fig, f"{pasta}/04_confianca.png")

        print(f"Relatório salvo: {pasta}/")
        return classe, confianca, pasta
