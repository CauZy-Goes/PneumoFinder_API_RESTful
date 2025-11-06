import os
import uuid
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

class DetectorDePneumoniaService:
    def __init__(self, caminho_modelo):
        self.model = load_model(caminho_modelo)
        self.img_size = (224, 224)

        # === Identifica a última camada Conv2D dentro da ResNet50 ===
        base_model = self.model.layers[0]
        self.last_conv_layer = None
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                self.last_conv_layer = layer
                break

        if self.last_conv_layer is None:
            raise ValueError("Nenhuma camada Conv2D encontrada na ResNet50!")

    def preprocessar_imagem(self, caminho_imagem):
        """Carrega e prepara a imagem para o modelo."""
        img = tf.keras.preprocessing.image.load_img(caminho_imagem, target_size=self.img_size)
        img_arr = tf.keras.preprocessing.image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = tf.keras.applications.resnet50.preprocess_input(img_arr)
        return img_arr
        
    def gerar_gradcam(self, img_arr, intensidade=0.7):
        """
        Gera o mapa Grad-CAM funcional e visível.
        Corrige a detecção da última camada convolucional e garante contraste.
        """

        # Identifica o modelo base (ResNet50) dentro do modelo composto
        base_model = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break

        if base_model is None:
            base_model = self.model

        # Garante que existe camada Conv2D
        last_conv_layer = None
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            raise ValueError("Nenhuma camada Conv2D encontrada para Grad-CAM.")

        grad_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer.output, base_model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_arr)
            class_channel = predictions[:, 0]

        grads = tape.gradient(class_channel, conv_outputs)
        grads = tf.where(tf.math.is_nan(grads), 0.0, grads)
        grads = tf.where(tf.math.is_inf(grads), 0.0, grads)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))

        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)[0]
        cam = np.maximum(cam, 0)
        cam = cam / (np.max(cam) + 1e-8)
        cam = cv2.resize(cam, self.img_size)

        # Aumenta contraste e visibilidade
        cam = np.power(cam, 1.2)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


    def diagnosticar_com_explicacao(self, caminho_imagem):
        """Executa o diagnóstico e gera as explicações visuais + PDF."""
        img_arr = self.preprocessar_imagem(caminho_imagem)

        pred = self.model.predict(img_arr, verbose=0)[0][0]
        classe = "PNEUMONIA" if pred > 0.5 else "NORMAL"
        confianca = pred if pred > 0.5 else 1 - pred

        cam = self.gerar_gradcam(img_arr)

        original = cv2.imread(caminho_imagem)
        original = cv2.resize(original, self.img_size)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        folder_name = f"{os.path.splitext(os.path.basename(caminho_imagem))[0]}_{uuid.uuid4().hex[:5]}"
        output_dir = os.path.join("relatorios", folder_name)
        os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(f"{output_dir}/original.jpg", original)
        cv2.imwrite(f"{output_dir}/heatmap.jpg", heatmap)
        cv2.imwrite(f"{output_dir}/gradcam_overlay.jpg", overlay)

        self._gerar_pdf(classe, confianca, output_dir)
        return classe, confianca, output_dir

    def _gerar_pdf(self, classe, confianca, output_folder):
        """Gera o relatório PDF com diagnóstico e Grad-CAM."""
        nome_pdf = os.path.join(output_folder, "relatorio.pdf")
        doc = SimpleDocTemplate(nome_pdf)
        styles = getSampleStyleSheet()

        elementos = [
            Paragraph("Relatório de Diagnóstico de Pneumonia com IA", styles["Title"]),
            Spacer(1, 20),
            Paragraph(f"Diagnóstico: <b>{classe}</b>", styles["Normal"]),
            Paragraph(f"Confiança: {confianca:.2%}", styles["Normal"]),
            Paragraph(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Normal"]),
            Spacer(1, 20),
            Paragraph("Mapa de ativação (Grad-CAM):", styles["Heading3"]),
            Image(os.path.join(output_folder, "gradcam_overlay.jpg"), width=300, height=300)
        ]

        doc.build(elementos)
