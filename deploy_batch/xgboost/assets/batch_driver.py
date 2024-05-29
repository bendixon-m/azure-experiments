
import os
import logging
import json
import numpy as np
import xgboost as xgb
import pandas as pd
from typing import List

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "xgboost_model.json"
    )
    model = xgb.Booster()
    model.load_model(model_path)
    logging.info("Init complete")


def run(mini_batch: List[str]) -> pd.DataFrame:
    """
    This function is called on each batch run to perform the actual scoring/prediction.
    In the example we loop through the batch and extract the data from the json input and call the xgboost predict, 
    then return the max class method and return the result back.
    """
    print(f"Executing run method over batch of {len(mini_batch)} files.")
    results = []
    for file_path in mini_batch:
        with open(file_path, 'r') as f:
            file = json.load(f)
            data = file["data"]
            raw_data = np.array(data)
            raw_data = raw_data.reshape(1, -1)
            input_data = xgb.DMatrix(raw_data) 
            raw_preds = model.predict(input_data)
            result = np.argmax(raw_preds, axis=1)
            results.append(result.tolist())

    return pd.DataFrame(results)

