# service/pneumonia_service.py
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


class DetectorDePneumoniaService:
    """
    Serviço para diagnóstico de pneumonia com Grad-CAM de alta qualidade.
    Saída: 4 imagens PNG em pasta por imagem (sem PDF).
    """
    def __init__(self, caminho_modelo, tamanho_img=(224, 224)):
        self.tamanho_img = tamanho_img
        self.model = load_model(caminho_modelo)

        # Forçar inicialização do modelo (resolve "never been called")
        dummy_input = tf.zeros((1, *tamanho_img, 3))
        _ = self.model(dummy_input, training=False)  # Força build

        # Encontrar ResNet50 base
        self.base_model = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model) and 'resnet' in layer.name.lower():
                self.base_model = layer
                break
        if not self.base_model:
            raise ValueError("Base ResNet50 não encontrada no modelo!")

        # Última camada convolucional
        self.last_conv_layer = None
        for layer in reversed(self.base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                self.last_conv_layer = layer
                break
        if not self.last_conv_layer:
            raise ValueError("Nenhuma camada Conv2D encontrada!")

        print(f"Grad-CAM pronto: última conv → {self.last_conv_layer.name}")
        print(f"Modelo carregado: {caminho_modelo}")

    def _preprocessar(self, caminho_imagem):
        img = image.load_img(caminho_imagem, target_size=self.tamanho_img)
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = tf.keras.applications.resnet50.preprocess_input(arr.copy())
        return img, arr

    def _gradcam_alta_qualidade(self, img_array):
        # Modelo Grad-CAM: entrada → última conv + saída final
        grad_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=[self.last_conv_layer.output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            pred = predictions[0][0]
            class_score = pred if pred > 0.5 else 1 - pred

        # Gradientes
        grads = tape.gradient(class_score, conv_outputs)
        if grads is None:
            grads = tf.zeros_like(conv_outputs)

        # Ponderação por canal
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(weights[None, None, :] * conv_outputs[0], axis=-1).numpy()

        # ReLU + normalização
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        # Super-resolução + suavização profissional
        h, w = self.tamanho_img
        cam = cv2.resize(cam, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=2.5)
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)

        # Reforço de contraste (gamma) + normalização final
        cam = np.power(cam, 0.7)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def _overlay_profissional(self, img_bgr, cam, alpha=0.65):
        # Heatmap com JET: vermelho em alta ativação
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]), cv2.INTER_CUBIC)
        heatmap = cv2.GaussianBlur(heatmap, (3, 3), 0)
        return cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)

    def diagnosticar_com_explicacao(self, caminho_imagem, pasta_saida="relatorios"):
        nome = os.path.splitext(os.path.basename(caminho_imagem))[0]
        pasta = os.path.join(pasta_saida, nome)
        os.makedirs(pasta, exist_ok=True)

        img_pil, img_arr = self._preprocessar(caminho_imagem)
        pred = self.model.predict(img_arr, verbose=0)[0][0]
        classe = "PNEUMONIA" if pred > 0.5 else "NORMAL"
        confianca = pred if pred > 0.5 else 1 - pred

        print(f"{nome}: {classe} ({confianca:.1%})")

        # Grad-CAM
        cam = self._gradcam_alta_qualidade(img_arr)
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        overlay = self._overlay_profissional(img_bgr, cam)

        def salvar(fig, caminho, dpi=300):
            fig.savefig(caminho, dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.1)
            plt.close(fig)

        # 01 Original
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_pil)
        ax.set_title(f"Original\n{classe} ({confianca:.1%})", fontsize=16, pad=20, weight='bold')
        ax.axis('off')
        salvar(fig, f"{pasta}/01_original.png")

        # 02 Grad-CAM
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cam, cmap='jet', vmin=0, vmax=1)
        ax.set_title("Grad-CAM (Ativação)", fontsize=16, pad=20, weight='bold')
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Intensidade", rotation=270, labelpad=15, fontsize=12)
        salvar(fig, f"{pasta}/02_gradcam.png")

        # 03 Overlay
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title("Foco do Modelo", fontsize=16, pad=20, weight='bold')
        ax.axis('off')
        salvar(fig, f"{pasta}/03_overlay.png")

        # 04 Confiança
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(['NORMAL', 'PNEUMONIA'], [1 - pred, pred],
                      color=['#2ca02c', '#d62728'], edgecolor='black', linewidth=1.2)
        ax.set_ylim(0, 1)
        ax.set_title("Confiança", fontsize=16, weight='bold')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.03, f'{h:.1%}',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        salvar(fig, f"{pasta}/04_confianca.png")

        print(f"Relatório salvo em: {pasta}/")
        return classe, confianca, pasta