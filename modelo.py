from tensorflow.keras.models import load_model
modelo = load_model("models/best_model.keras")
modelo.summary()
