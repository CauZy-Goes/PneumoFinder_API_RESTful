import os
import uuid
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

class DetectorDePneumoniaAvancado:
    def __init__(self, caminho_modelo):
        # Carrega o modelo
        self.model = load_model(caminho_modelo)
        self.img_size = (224, 224)

        # Detecta a última camada Conv2D dentro do modelo (funciona com Sequential + ResNet50)
        self.last_conv_layer = None
        for layer in reversed(self.model.layers):
            # se for modelo dentro do Sequential, checa suas camadas
            if isinstance(layer, tf.keras.Model):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        self.last_conv_layer = sub_layer
                        break
            elif isinstance(layer, tf.keras.layers.Conv2D):
                self.last_conv_layer = layer
            if self.last_conv_layer:
                break

        if self.last_conv_layer is None:
            raise ValueError("Nenhuma camada Conv2D encontrada no modelo!")

    def preprocessar(self, img_path):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.img_size)
        img_arr = tf.keras.preprocessing.image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        # Preprocessamento padrão do ResNet50
        img_arr = tf.keras.applications.resnet50.preprocess_input(img_arr)
        return img_arr

    def gerar_gradcam(self, img_arr):
        # Chama o modelo para inicializar tensores
        _ = self.model(img_arr)

        # Detecta a última camada Conv2D
        last_conv = None
        for layer in reversed(self.model.layers):
            # se for modelo dentro do Sequential, checa suas camadas
            if isinstance(layer, tf.keras.Model):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        last_conv = sub_layer
                        break
            elif isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer
            if last_conv:
                break

        if last_conv is None:
            raise ValueError("Nenhuma camada Conv2D encontrada no modelo!")

        # Modelo intermediário Grad-CAM
        grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[last_conv.output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_arr)
            loss = predictions[:, 0]  # classificação binária

        grads = tape.gradient(loss, conv_outputs)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
        cam = np.maximum(cam[0], 0)
        cam = cv2.resize(cam.numpy(), self.img_size)
        cam = cam / (cam.max() + 1e-8)
        return cam


    def diagnosticar_com_explicacao(self, caminho_imagem):
        img_arr = self.preprocessar(caminho_imagem)
        pred = self.model.predict(img_arr)[0][0]
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
        nome_pdf = f"{output_folder}/relatorio.pdf"
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
            Image(f"{output_folder}/gradcam_overlay.jpg", width=300, height=300),
        ]

        doc.build(elementos)
