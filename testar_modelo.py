import os
import pneumo_finder_service as pf

# Caminhos base
CAMINHO_MODELO = "best_model.keras"
CAMINHO_IMGS = "imgs\\"

# Inicializa detector
detector = pf.DetectorDePneumonia(CAMINHO_MODELO)

def menu():
    while True:
        print("\n=== MENU ===")
        print("1. Diagnosticar uma imagem")
        print("2. Diagnosticar todas as imagens da pasta 'imgs/'")
        print("0. Sair")

        escolha = input("Escolha uma opção: ")

        if escolha == "1":
            nome_img = input("Digite o nome da imagem (ex: 1_normal1.jpeg): ")
            caminho_img = CAMINHO_IMGS + nome_img
            if os.path.exists(caminho_img):
                detector.diagnosticar_imagem(caminho_img)
            else:
                print("❌ Imagem não encontrada.")
        elif escolha == "2":
            detector.diagnosticar_pasta(CAMINHO_IMGS)
        elif escolha == "0":
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    menu()
