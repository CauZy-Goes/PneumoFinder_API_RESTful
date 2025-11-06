import os
from service.pneumonia_service import DetectorDePneumoniaService

if __name__ == "__main__":
    modelo_path = "models/best_model.keras"
    imagens_dir = "imgs"  # pasta com todas as imagens
    detector = DetectorDePneumoniaService(modelo_path)

    # Cria pasta de saída geral
    output_base_dir = "relatorios_todos"
    os.makedirs(output_base_dir, exist_ok=True)

    # Processa todas as imagens da pasta
    for nome_arquivo in os.listdir(imagens_dir):
        if nome_arquivo.lower().endswith((".png", ".jpg", ".jpeg")):
            caminho_imagem = os.path.join(imagens_dir, nome_arquivo)
            print(f"Processando: {caminho_imagem} ...")

            try:
                classe, confianca, relatorio = detector.diagnosticar_com_explicacao(caminho_imagem)
                print(f"→ Diagnóstico: {classe}, Confiança: {confianca:.2%}")
                print(f"→ Relatório salvo em: {relatorio}\n")
            except Exception as e:
                print(f"Erro ao processar {nome_arquivo}: {e}\n")
