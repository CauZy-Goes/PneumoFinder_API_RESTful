# testar_modelo.py
import os
from service.pneumonia_service import DetectorDePneumoniaService

if __name__ == "__main__":
    detector = DetectorDePneumoniaService("models/best_model_d1.keras")
    
    pasta_imgs = "imgs"
    formatos_validos = (".png", ".jpg", ".jpeg", ".bmp")
    
    # percorre todas as imagens da pasta
    for arquivo in os.listdir(pasta_imgs):
        if arquivo.lower().endswith(formatos_validos):
            caminho = os.path.join(pasta_imgs, arquivo)
            classe, confianca, pasta = detector.diagnosticar_com_explicacao(caminho)
            print(f"\nDiagnóstico: {classe} ({confianca:.1%})")
            print(f"Imagens salvas em: {pasta}")
