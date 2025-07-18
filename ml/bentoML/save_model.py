import bentoml
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
# Entraîne ton modèle ici si tu veux...

bentoml.sklearn.save_model("mon_modele", model)
print("Modèle sauvegardé avec succès sous le nom 'mon_modele'.")
