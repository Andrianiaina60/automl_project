import pickle
import os

model_path = os.path.abspath("best_model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Fichier introuvable : {model_path}")
with open(model_path, "rb") as f:
    model = pickle.load(f)
print(f"✅ Modèle chargé avec succès depuis : {model_path}")