import bentoml
from bentoml.io import JSON
import pandas as pd

svc = bentoml.Service("svc")

@svc.api(input=JSON(), output=JSON())
async def predict(input_data):
    model = bentoml.sklearn.get("regressor:eeznxfdjiwc2saav").to_runner()
    df = pd.DataFrame([input_data])
    prediction = model.predict.run(df)
    return {"prediction": prediction.tolist()}


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


