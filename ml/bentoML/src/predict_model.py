import os
import pandas as pd
from pycaret.classification import load_model, predict_model

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_NAME = "best_model"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)
INPUT_FILE = os.path.join(BASE_DIR, "../data/employees.xlsx")
OUTPUT_FILE = os.path.join(BASE_DIR, "predictions.xlsx")

print("📦 Chargement du modèle...")
model = load_model(MODEL_PATH)

print("📄 Chargement des données à prédire...")
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Fichier introuvable : {INPUT_FILE}")
new_data = pd.read_excel(INPUT_FILE)

print("🧹 Nettoyage des données...")
new_data.dropna(inplace=True)
new_data.drop_duplicates(inplace=True)
id_cols = [col for col in new_data.columns if 'id' in col.lower()]
new_data.drop(columns=id_cols, errors='ignore', inplace=True)

print("🤖 Prédiction en cours...")
predictions = predict_model(model, data=new_data)

print(f"💾 Sauvegarde des prédictions dans {OUTPUT_FILE}...")
predictions.to_excel(OUTPUT_FILE, index=False)

print("✅ Prédictions terminées avec succès.")