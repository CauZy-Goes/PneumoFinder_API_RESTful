import os
import shutil
import zipfile
from flask import Flask, request, jsonify, send_file
from service.pneumonia_service import DetectorDePneumoniaService
from tempfile import mkdtemp

app = Flask(__name__)

# Inicializa o detector (carrega o modelo na memória uma única vez)
DETECTOR = DetectorDePneumoniaService(caminho_modelo="models/best_model_d1.keras")


@app.route("/diagnosticar", methods=["POST"])
def diagnosticar():
    if "file" not in request.files:
        return jsonify({"erro": "Nenhum arquivo enviado"}), 400

    file = request.files["file"]

    # Cria pasta temporária
    temp_dir = mkdtemp(prefix="pneumofinder_")
    temp_path = os.path.join(temp_dir, file.filename)

    # Salva arquivo temporário
    file.save(temp_path)

    try:
        # Chama serviço de diagnóstico
        classe, confianca, pasta_relatorio = DETECTOR.diagnosticar_com_explicacao(temp_path)

        # Caminho do arquivo ZIP final
        zip_path = os.path.join(temp_dir, "relatorio_pneumofinder.zip")

        # Cria o ZIP com todas as imagens geradas
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for nome_arquivo in os.listdir(pasta_relatorio):
                caminho_completo = os.path.join(pasta_relatorio, nome_arquivo)
                zipf.write(caminho_completo, arcname=nome_arquivo)

        # Retorna o ZIP como download
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=f"relatorio_{classe.lower()}.zip",
            mimetype="application/zip"
        )

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

    finally:
        # Limpa o arquivo original após gerar o relatório
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Não precisa remover temp_dir ainda, pois o ZIP é retornado de lá
    # o sistema operacional vai limpar pastas temporárias automaticamente depois


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
