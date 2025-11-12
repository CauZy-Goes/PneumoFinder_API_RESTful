# pneumonia_service/main_service.py
import os
import cv2
from .diagnosticador import Diagnosticador
from .gradcam import GradCAM
import matplotlib.pyplot as plt
import numpy as np

class PneumoniaService:
    def __init__(self, caminho_modelo, tamanho_img=(224,224), last_conv_layer_name="conv5_block3_3_conv"):
        self.diagnosticador = Diagnosticador(caminho_modelo, tamanho_img)
        self.tamanho_img = tamanho_img
        # GradCAM com base no modelo do Diagnosticador
        self.gradcam = GradCAM(self.diagnosticador.model, last_conv_layer_name, tamanho_img)

    def rodar(self, caminho_imagem, pasta_saida="relatorios"):
        nome = os.path.splitext(os.path.basename(caminho_imagem))[0]
        pasta = os.path.join(pasta_saida, nome)
        os.makedirs(pasta, exist_ok=True)

        img_pil, arr = self.diagnosticador.preprocessar(caminho_imagem)
        classe, confianca = self.diagnosticador.diagnosticar(arr)
        print(f"{nome}: {classe} ({confianca:.1%})")

        cam = self.gradcam.calcular_cam(arr)
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        overlay = self.gradcam.overlay(img_bgr, cam, alpha=0.7)

        # Salvar imagens
        def salvar(fig, path):
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(img_pil)
        ax.set_title(f"Original\n{classe} ({confianca:.1%})", fontsize=14, pad=15, weight='bold')
        ax.axis('off')
        salvar(fig, f"{pasta}/01_original.png")

        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(cam, cmap='hot', vmin=0, vmax=1)
        ax.set_title("Grad-CAM", fontsize=14, pad=15, weight='bold')
        ax.axis('off')
        salvar(fig, f"{pasta}/02_gradcam.png")

        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title("Regiões de Pneumonia", fontsize=14, pad=15, weight='bold')
        ax.axis('off')
        salvar(fig, f"{pasta}/03_overlay.png")

        fig, ax = plt.subplots(figsize=(5,4))
        bars = ax.bar(['NORMAL','PNEUMONIA'], [1-confianca, confianca], color=['#2ca02c','#d62728'], edgecolor='black')
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.03, f'{b.get_height():.1%}', ha='center', fontweight='bold', fontsize=12)
        ax.set_ylim(0,1)
        ax.set_title("Confiança", fontsize=14, weight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        salvar(fig, f"{pasta}/04_confianca.png")

        print(f"Relatório salvo: {pasta}/")
        return classe, confianca, pasta
