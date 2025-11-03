# menu.py
import os
from service import pneumonia_service as pf

CAMINHO_MODELO = "models\\best_model.keras"
CAMINHO_IMGS = "imgs\\"
PASTA_RELATORIOS = "relatorios"

detector = pf.DetectorDePneumoniaAvancado(CAMINHO_MODELO)

def menu():
    while True:
        print("\n" + "="*60)
        print("   DETECTOR DE PNEUMONIA COM IA + XAI (GRAD-CAM)")
        print("="*60)
        print("1. Diagnosticar uma imagem (com pasta XAI)")
        print("3. Abrir pasta de relatórios")
        print("0. Sair")
        print("-"*60)

        escolha = input("Escolha: ").strip()

        if escolha == "1":
            nome = input("Nome da imagem (ex: person72_bacteria_352.jpeg): ").strip()
            caminho = os.path.join(CAMINHO_IMGS, nome)
            if os.path.exists(caminho):
                print("\nProcessando...")
                classe, conf, rel = detector.diagnosticar_com_explicacao(caminho)
                print(f"\nDIAGNÓSTICO: {classe} ({conf:.1%})")
                print(f"Pasta XAI: {os.path.dirname(rel)}/")
                input("\nENTER para continuar...")
            else:
                print("Imagem não encontrada!")
                input("ENTER...")

        elif escolha == "3":
            caminho_abs = os.path.abspath(PASTA_RELATORIOS)
            if os.path.exists(caminho_abs):
                os.startfile(caminho_abs) if os.name == 'nt' else os.system(f"xdg-open {caminho_abs}")
            else:
                print("Pasta de relatórios ainda não criada.")
            input("ENTER...")

        elif escolha == "0":
            print("Saindo... Obrigado!")
            break
        else:
            print("Opção inválida!")
            input("ENTER...")

if __name__ == "__main__":
    menu()