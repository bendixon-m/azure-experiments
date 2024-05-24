# scoring script https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&tabs=python#understand-the-scoring-script

import os
import logging
import json
import numpy as np
import joblib
import xgboost as xgb


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



def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the xgboost predict, then return the max class
    method and return the result back
    """
    logging.info("model 1: request received")
    data = json.loads(raw_data)["data"]
    data = np.array(data)

    data = xgb.DMatrix(data)
    raw_preds = model.predict(data)
    result = np.argmax(raw_preds, axis=1)
    logging.info("Request processed")
    return result.tolist()