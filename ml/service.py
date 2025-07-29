import bentoml
from bentoml.io import JSON
import pandas as pd
svc = bentoml.Service("svc")
@svc.api(input=JSON(), output=JSON())
async def predict(input_data):
    model = bentoml.sklearn.get("regressor_model:l4m5badjisj7gaav").to_runner()
    df = pd.DataFrame([input_data])
    prediction = model.predict.run(df)
    return {"prediction": prediction.tolist()}
