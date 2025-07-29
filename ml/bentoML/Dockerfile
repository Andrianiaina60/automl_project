FROM python:3.11-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Installer les packages Python
RUN pip install --no-cache-dir \
    bentoml==1.3.22 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scikit-learn==1.5.2 \
    pydantic==2.9.2

# Créer le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY bentofile.yaml ./
COPY service.py ./
COPY requirements.txt ./

# IMPORTANT: Utiliser 'build' et non 'import'
RUN bentoml build .

# Exposer le port
EXPOSE 3000

# Démarrer le service
CMD ["bentoml", "serve", "--production", "--port", "3000"]