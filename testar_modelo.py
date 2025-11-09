from service.pneumonia_service import DetectorDePneumoniaService

if __name__ == "__main__":
    detector = DetectorDePneumoniaService("models/best_model.keras")
    classe, confianca, pasta = detector.diagnosticar_com_explicacao("imgs/person75_bacteria_365.jpeg")

    print(f"\nDiagnóstico: {classe} ({confianca:.1%})")
    print(f"Imagens salvas em: {pasta}")

    detector = DetectorDePneumoniaService("models/best_model.keras")
    classe, confianca, pasta = detector.diagnosticar_com_explicacao("imgs/person72_bacteria_352.jpeg")

    print(f"\nDiagnóstico: {classe} ({confianca:.1%})")
    print(f"Imagens salvas em: {pasta}")

    detector = DetectorDePneumoniaService("models/best_model.keras")
    classe, confianca, pasta = detector.diagnosticar_com_explicacao("imgs/person74_bacteria_361.jpeg")

    print(f"\nDiagnóstico: {classe} ({confianca:.1%})")
    print(f"Imagens salvas em: {pasta}")


