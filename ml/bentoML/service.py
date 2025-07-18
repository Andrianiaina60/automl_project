import bentoml
from bentoml.io import JSON

svc = bentoml.Service("svc")

@svc.api(input=JSON(), output=JSON())
def predict(input_data):
    return {"result": "ok"}
