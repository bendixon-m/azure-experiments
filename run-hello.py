# Adapted from get-started/run-hello.py for SDK v2

from azure.ai.ml.entities import Environment
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command


credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential)

custom_job_env = Environment(
    name="hello-world-env",
    description="Custom environment for basic hello world job",
    conda_file="azureml_envs/hello-world.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)
custom_job_env = ml_client.environments.create_or_update(custom_job_env)

print(
    f"Environment with name {custom_job_env.name} is registered to workspace, the environment version is {custom_job_env.version}"
)

job = command(
    code="./src/",
    command="python hello.py",
    display_name="hello-world-sdk-v2",
    environment="hello-world-env@latest",
    compute="ben-small-test" # assigning serverless at the moment gives an error on permissions to access docker registry
)

ml_client.create_or_update(job)
