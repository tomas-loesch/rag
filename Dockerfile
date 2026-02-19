FROM python:3.11-slim

# Definir o diretório de trabalho dentro do container
WORKDIR /app

# Instalar dependências do sistema necessárias para o ChromaDB e PDF
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar o arquivo de requisitos primeiro (melhora o cache do Docker)
COPY requirements.txt .

# Instalar as bibliotecas do Python
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Comando para rodar a aplicação
CMD ["python", "main.py"]