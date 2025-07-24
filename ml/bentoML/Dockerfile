FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Mettre à jour pip
RUN pip install --upgrade pip

# Copier le fichier requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers
COPY . .

# Exposer le port
EXPOSE 3000

# Commande par défaut pour BentoML 1.3
CMD ["bentoml", "serve", "service.py:MyService", "--host", "0.0.0.0", "--port", "3000"]