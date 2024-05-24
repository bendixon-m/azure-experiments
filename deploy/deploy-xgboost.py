# Useful links
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-environments-v2?view=azureml-api-2&tabs=cli#use-environments-for-training

import uuid
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment
from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential)

REGISTERED_MODEL_NAME = "xgboost_model"

def upload_model():
    xgboost_model = Model(
        path="deploy/assets/xgboost_model.json",
        type=AssetTypes.CUSTOM_MODEL,
        name=REGISTERED_MODEL_NAME,
        description="Model created from local files.",
    )
    ml_client.models.create_or_update(xgboost_model)


def get_latest_model():
    latest_model_version = max(
        [int(m.version) for m in ml_client.models.list(name=REGISTERED_MODEL_NAME)]
    )
    print(f"Latest model version: {latest_model_version}")
    latest_model = ml_client.models.get(name=REGISTERED_MODEL_NAME, version=latest_model_version)
    return latest_model


def register_environment():

    
    env = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file="./azureml_envs/xgboost.yaml",
        name="xgboost-inference-env",
        description="Environment for xgboost inference created from a Docker image plus Conda environment.",
    )
    ml_client.environments.create_or_update(env)
    return env


def create_endpoint():
    online_endpoint_name = "xgboost-model-endpoint-" + str(uuid.uuid4())[:8]
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        description="this is an online endpoint for the xgboost model and it's great!",
        auth_mode="key",
        tags={
            "training_dataset": "sklearn_iris",
        },
    )
    endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result() # takes approximately 2 minutes.
    endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)
    print(
        f'Endpoint "{endpoint.name}" with provisioning state "{endpoint.provisioning_state}" is retrieved'
    )
    return endpoint.name


def create_deployment(endpoint, env, model, online_endpoint_name, local=True):

    """Creates a deployment of the specified endpoint, defaulting to local. 
    """

    blue_deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=online_endpoint_name,
        model=model,
        instance_type="Standard_DS3_v2",
        instance_count=1,
        code_configuration=CodeConfiguration(
            code="deploy/", 
            scoring_script="score.py"
        ),
        environment=env
    )

    print('Create online deployment (blue)')
    
    blue_deployment = ml_client.online_deployments.begin_create_or_update(
        deployment=blue_deployment, local=local
    ).result()

    # blue deployment takes 100% traffic
    # expect the deployment to take approximately 8 to 10 minutes.
    print('Route all traffic to blue deployment')
    endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def print_endpoint_metadata(endpoint, local=True):
    endpoint = ml_client.online_endpoints.get(name=online_endpoint_name, local=local)
    print(
        f"Name: {endpoint.name}\nStatus: {endpoint.provisioning_state}\nDescription: {endpoint.description}"
    )
    print(endpoint.traffic)
    print(endpoint.scoring_uri)


if __name__ == "__main__":
    #upload_model()
    local_model = Model(path="deploy/assets/xgboost_model.json")
    env = register_environment()
    online_endpoint_name = "xgboost-model-endpoint-13cb9b54"
    endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)
    latest_model = get_latest_model()


    create_deployment(endpoint, env, local_model, online_endpoint_name, local=True)