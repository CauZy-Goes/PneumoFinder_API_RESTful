from service.pneumonia_service import DetectorDePneumoniaService

if __name__ == "__main__":
    detector = DetectorDePneumoniaService("models/best_model.keras")
    classe, confianca, pasta = detector.diagnosticar_com_explicacao("imgs/pneumo_web_2.png")

    print(f"\nDiagnóstico: {classe} ({confianca:.1%})")
    print(f"Imagens salvas em: {pasta}")