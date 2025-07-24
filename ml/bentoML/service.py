
# from bentoml import Service
# from bentoml.io import JSON
# from pydantic import BaseModel

# class InputData(BaseModel):
#     message: str

# svc = Service("svc")

# @svc.api(input=JSON(pydantic_model=InputData), output=JSON())
# async def predict(input_data: InputData):
#     return {"result": f"Received: {input_data.message}"}

# from bentoml import Service
# from bentoml.io import Multipart, File, JSON
# import pandas as pd
# import io

# svc = Service("svc")

# @svc.api(input=Multipart(file=File()), output=JSON())
# def predict(file: bytes):
#     df = pd.read_excel(io.BytesIO(file))
#     return {
#         "shape": df.shape,
#         "columns": df.columns.tolist(),
#         "preview": df.head(3).to_dict(orient="records")
#     }


import bentoml
import pandas as pd
from bentoml.io import File, JSON

# Charger le modèle
model_runner = bentoml.sklearn.get("best_meteo_model:latest").to_runner()
svc = bentoml.Service("meteo_prediction_service", runners=[model_runner])

@svc.api(input=File(), output=JSON())
async def predict(file):
    df = pd.read_excel(file)

    # Assurer que les colonnes sont bien présentes
    if not all(col in df.columns for col in ["temperature", "humidite", "vent"]):
        return {"error": "Colonnes manquantes"}

    # Retirer la dernière ligne si on ne peut pas prédire après
    features = df[["temperature", "humidite", "vent"]][:-1]

    # Prédiction de la température du lendemain
    preds = await model_runner.async_run(features)

    # Associer chaque prédiction à une date suivante
    dates = pd.to_datetime(df["date"][:-1])
    pred_dates = dates + pd.Timedelta(days=1)

    result = [{"date": d.strftime("%Y-%m-%d"), "predicted_temperature": t} for d, t in zip(pred_dates, preds)]

    return {"predictions": result}
