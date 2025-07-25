from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from service import pneumonia_service as pf
from service import pulmao_service as pulm


app = Flask(__name__)
CORS(app)

# Inicializa os dois detectores
detector_pneumonia = pf.DetectorDePneumonia("models/pneumonia_model.keras")
detector_pulmao = pulm.DetectorDePulmao("models/pulmao_model.keras")

@app.route("/diagnosticar_pneumonia", methods=["POST"])
def diagnosticar_pneumonia():
    if "imagem" not in request.files:
        return jsonify({"erro": "Nenhuma imagem enviada."}), 400

    imagem = request.files["imagem"]
    caminho_temp = os.path.join("temp", imagem.filename)
    imagem.save(caminho_temp)

    try:
        classe, confianca = detector_pneumonia.diagnosticar_imagem(caminho_temp)
        os.remove(caminho_temp)
        response = {
            "classe": str(classe),
            "confianca": float(round(confianca.item(), 2))
        }
        print("Resposta gerada para o front:", response)
        return jsonify(response)
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/verificar_pulmao", methods=["POST"])
def verificar_pulmao():
    if "imagem" not in request.files:
        return jsonify({"erro": "Nenhuma imagem enviada."}), 400

    imagem = request.files["imagem"]
    caminho_temp = os.path.join("temp", imagem.filename)
    imagem.save(caminho_temp)

    try:
        classe, confianca = detector_pulmao.detectar_imagem(caminho_temp)
        os.remove(caminho_temp)
        response = {
            "classe": str(classe),
            "confianca": float(round(confianca.item(), 2))
        }
        print("Resposta gerada para o front (pulmão):", response)
        return jsonify(response)
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/diagnostico_completo", methods=["POST"])
def diagnostico_completo():
    if "imagem" not in request.files:
        return jsonify({"erro": "Nenhuma imagem enviada."}), 400

    imagem = request.files["imagem"]
    caminho_temp = os.path.join("temp", imagem.filename)
    imagem.save(caminho_temp)

    try:
        # Primeiro: verifica se é um pulmão
        classe_pulmao, confianca_pulmao = detector_pulmao.detectar_imagem(caminho_temp)

        if classe_pulmao != "PULMÃO":
            os.remove(caminho_temp)
            return jsonify({
                "classe_pulmao": "NÃO É PULMÃO",
                "confianca": float(round(confianca_pulmao.item(), 2))
            })

        # Segundo: diagnostica pneumonia
        classe_pneumonia, confianca_pneumonia = detector_pneumonia.diagnosticar_imagem(caminho_temp)
        os.remove(caminho_temp)

        response = {
            "classe_pulmao": "PULMÃO",
            "confianca_pulmao": float(round(confianca_pulmao.item(), 2)),
            "classe_pneumonia": classe_pneumonia,
            "confianca_pneumonia": float(round(confianca_pneumonia.item(), 2))
        }
        print("Resposta completa para o front:", response)
        return jsonify(response)

    except Exception as e:
        return jsonify({"erro": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("temp", exist_ok=True)
    app.run(debug=True, port=5001)
