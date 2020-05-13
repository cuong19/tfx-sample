import json
from typing import List

import requests
import numpy as np

from imagenet_inference import get_image, preprocess


def rest_infer(instances: List,
               model_name='adder',
               host='localhost',
               port=8501,
               signature_name="serving_default"
               ):
    data = json.dumps({
        "signature_name": signature_name,
        "instances": instances
    })

    headers = {"content-type": "application/json"}
    json_response = requests.post(
        'http://{}:{}/v1/models/{}:predict'.format(host, port, model_name),
        data=data,
        headers=headers
    )
    print(json_response.json())


if __name__ == "__main__":
    img = np.load('kitten.npy').tolist()
    rest_infer([img])
