import json
from typing import List

import grpc
import requests
import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2


def rest_infer(instances: List,
               model_name='adder',
               host='localhost',
               port=8501,
               signature_name='serving_default'
               ):
    data = json.dumps({
        "signature_name": signature_name,
        "instances": instances
    })

    headers = {"content-type": "application/json"}
    json_response = requests.post(
        f'http://{host}:{port}/v1/models/{model_name}:predict',
        data=data,
        headers=headers
    )
    print(json_response.json())


def grpc_infer(instances: List,
               model_name='adder',
               host='localhost',
               port=8500,
               signature_name='serving_default'):
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name
    request.inputs["x"].CopyFrom(tf.make_tensor_proto(instances))
    result = stub.Predict(request)
    result = result.outputs["output_0"]
    print(result)


if __name__ == "__main__":
    val = [1.0, 2.0, 3.0]
    rest_infer(val)
    grpc_infer(val)
