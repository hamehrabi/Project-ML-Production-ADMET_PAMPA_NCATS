from monitoring.prometheus import REQUEST_TIME, start_http_server, COUNT_p, COUNT_r, COUNT_w
from aws_model_registry.model_registry import ModelRegistryConnection
from data_preprocessing_service.inference_loader import ObjectLoader
from utils.utils import read_config
from fastapi import FastAPI, Request
import uvicorn
import time

app = FastAPI()

model = None


class PrepareEndpoints:
    def __init__(self):
        self.config = read_config()
        self.model_registry = self.config["model_registry"]["bucket_name"]
        self.zip_files = self.config["model_registry"]["zip_files"]
        self.package_name = self.config["model_registry"]["package_name"]

    def inference_object_loader(self) -> None:
        global model

        registry = ModelRegistryConnection(self.model_registry,
                                           self.zip_files,
                                           self.package_name)
        registry.get_package_from_prod()
        loader = ObjectLoader()
        prod_objects = loader.load_objects()

        model = prod_objects["model"]


@REQUEST_TIME.time()
@app.get('/')
def invoke():
    COUNT_w.inc()
    return {"Response": "Hello world from Model Endpoint"}


@REQUEST_TIME.time()
@app.post('/predict')
async def predict(request: Request):
    query = await request.json()
    result = model.predict(query)
    result = {"Result": result.tolist()[0]}
    COUNT_p.inc()
    return result


@REQUEST_TIME.time()
@app.get('/reload')
def reload():
    executor = PrepareEndpoints()
    executor.inference_object_loader()
    COUNT_r.inc()
    return {"Response": "Updating Model In Prod"}


if __name__ == "__main__":
    executor = PrepareEndpoints()
    executor.inference_object_loader()
    start_http_server(5000)
    uvicorn.run(app, host="localhost", port=8081)
