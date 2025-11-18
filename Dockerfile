# Imagem base
FROM python:3.11-slim


# Instala dependências do sistema necessárias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements e instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o projeto (incluindo modelos)
COPY . .

# Expõe porta
EXPOSE 5000

# Comando de inicialização
CMD ["python", "api_flask.py"]
