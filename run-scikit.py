# run-scikit.py
# this uses SDK v2!

from azure.ai.ml.entities import Environment
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command


if __name__ == "__main__":
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential)

    custom_job_env = Environment(
        name="scikit-env",
        description="Custom environment for basic hello world job",
        conda_file="azureml_envs/scikit-env.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )

    custom_job_env = ml_client.environments.create_or_update(custom_job_env)

    print(
        f"Environment with name {custom_job_env.name} is registered to workspace, the environment version is {custom_job_env.version}"
    )

    job = command(
        code="./src/",
        command="python train-scikit.py",
        display_name="run-scitkit-training",
        environment="scikit-env@latest",
        compute="ben-small-test" # assigning serverless at the moment gives an error on permissions to access docker registry
    )

    ml_client.create_or_update(job)