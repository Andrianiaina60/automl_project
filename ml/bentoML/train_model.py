import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import bentoml

# Charger le fichier météo
df = pd.read_excel("meteo.xlsx")

# Caractéristiques et cible
X = df[["temperature", "humidite", "vent"]]   # données actuelles
y = df["temperature"].shift(-1).dropna()      # température du jour suivant

X = X[:-1]  # Car on a fait shift(-1)
y = y       # déjà bien

# Entraîner modèle
model = RandomForestRegressor()
model.fit(X, y)

# Sauvegarder dans BentoML
bentoml.sklearn.save_model("best_meteo_model", model)
