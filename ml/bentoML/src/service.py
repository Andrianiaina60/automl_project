# import bentoml
# from bentoml.io import PandasDataFrame
# import pandas as pd
# from pycaret.classification import predict_model

# try:
#     model_runner = bentoml.pycaret.get("employee_attrition_model:latest").to_runner()
# except Exception as e:
#     print(f"⚠️ Erreur lors du chargement avec bentoml.pycaret : {str(e)}")
#     try:
#         model = bentoml.models.get("employee_attrition_model:latest").load()
#         model_runner = bentoml.PicklableModel(model).to_runner()
#     except Exception as e2:
#         print(f"❌ Erreur lors du chargement avec bentoml.models : {str(e2)}")
#         raise

# svc = bentoml.Service("employee_attrition_service", runners=[model_runner])

# @svc.api(input=PandasDataFrame(), output=PandasDataFrame())
# async def predict(input_df: pd.DataFrame) -> pd.DataFrame:
#     input_df.dropna(inplace=True)
#     input_df.drop_duplicates(inplace=True)
#     id_cols = [col for col in input_df.columns if 'id' in col.lower()]
#     input_df.drop(columns=id_cols, errors='ignore', inplace=True)
#     predictions = await model_runner.predict_model.async_run(input_df)
#     return predictions[['prediction_label', 'prediction_score']]

import bentoml
from bentoml.io import JSON
import pandas as pd

# Charger le modèle depuis BentoML Model Store
model_ref = bentoml.sklearn.get("regressor_model:latest")
model_runner = model_ref.to_runner()

svc = bentoml.Service("employee_attrition_service", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
async def predict(input_data):
    df = pd.DataFrame([input_data])
    result = await model_runner.predict.async_run(df)
    return {"prediction": int(result[0])}
