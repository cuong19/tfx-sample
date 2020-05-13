import json

import requests


def rest_infer(x,
               model_name='adder',
               host='localhost',
               port=8501,
               signature_name="serving_default"
               ):
    data = json.dumps({
        "signature_name": signature_name,
        "instances": [x]
    })

    headers = {"content-type": "application/json"}
    json_response = requests.post(
        'http://{}:{}/v1/models/{}:predict'.format(host, port, model_name),
        data=data,
        headers=headers
    )
    print(json_response.json())


if __name__ == "__main__":
    rest_infer(1)