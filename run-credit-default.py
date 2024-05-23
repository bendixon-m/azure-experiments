# Uses SDK v2

import json
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment
from azure.ai.ml import command
from azure.ai.ml import Input

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential)

# Verify that the handle works correctly by printing workspace info
with open('config.json', 'r') as file:
    data = json.load(file)
    workspace_name = data.get('workspace_name', None)
ws = ml_client.workspaces.get(workspace_name)
print(ws.location, ":", ws.resource_group)

custom_env_name = "aml-scikit-learn_local"

custom_job_env = Environment(
    name=custom_env_name,
    description="Custom environment for Credit Card Defaults job",
    tags={"scikit-learn": "1.0.2"},
    conda_file="azureml_envs/credit-default.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)
custom_job_env = ml_client.environments.create_or_update(custom_job_env)

print(
    f"Environment with name {custom_job_env.name} is registered to workspace, the environment version is {custom_job_env.version}"
)

registered_model_name = "credit_defaults_model_ben_local"

job = command(
    inputs=dict(
        data=Input(
            type="uri_file",
            path="https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default_of_credit_card_clients.csv",
        ),
        test_train_ratio=0.2,
        learning_rate=0.25,
        registered_model_name=registered_model_name,
    ),
    code="./src/",  # location of source code
    command="python train-credit-default.py --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} --learning_rate ${{inputs.learning_rate}} --registered_model_name ${{inputs.registered_model_name}}",
    environment="aml-scikit-learn@latest",
    display_name="credit_default_prediction_ben", 
    compute='ben-small-test'
)

ml_client.create_or_update(job)