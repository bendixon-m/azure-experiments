"""Based on Azure docs: https://github.com/Azure/azureml-examples/tree/main/sdk/python/endpoints/batch/deploy-models/mnist-classifier"""
import logging
import random
import string
import time

from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import (
    BatchEndpoint,
    ModelBatchDeployment,
    ModelBatchDeploymentSettings,
    Model,
    Environment,
    AmlCompute,
    BatchRetrySettings,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction



class AzureMLBatchDeployment:

    """
    Sets up the resources required and deployment for a batch endpoint in Azure 
    ML workspace. Based on the Azure docs in the Azure/azureml-examples/ repo
    under sdk/python/endpoints/batch/deploy-models/mnist-classifier, adapted 
    for XGBoost. Requires the file `batch_driver.py`. 
    """

    def __init__(self):    
        self.credential = DefaultAzureCredential()
        self.ml_client = MLClient.from_config(self.credential)

    def create_batch_endpoint(self, model_name):
        self.endpoint_name = f"{model_name}-batch"
        allowed_chars = string.ascii_lowercase + string.digits
        endpoint_suffix = "".join(random.choice(allowed_chars) for x in range(5))
        self.endpoint_name = f"{self.endpoint_name}-{endpoint_suffix}"
        self.endpoint = BatchEndpoint(
            name=self.endpoint_name,
            description=f"A batch endpoint for returning {model_name} inferences"
        )
        self.ml_client.begin_create_or_update(self.endpoint).result()

    def use_existing_batch_endpoint(self, endpoint_name: str):
        self.endpoint_name = endpoint_name
        
    def register_model(self, model_name: str, model_local_path: str):
        xgboost_model = Model(
            path=model_local_path,
            type=AssetTypes.CUSTOM_MODEL,
            name=model_name,
            description=f"Model name {model_name} uploaded from local storage.",
        )
        self.ml_client.models.create_or_update(xgboost_model)
        self.model = self.ml_client.models.get(name=model_name, label="latest")

    def use_preregistered_model(self, model_name: str):
        self.model = self.ml_client.models.get(name=model_name, label="latest")
    
    def create_serverless_compute_cluster(self, compute_name: str):
        if not any(filter(lambda m: m.name == compute_name, 
                          self.ml_client.compute.list())):
            compute_cluster = AmlCompute(
                name=compute_name,
                description="CPU cluster compute",
                min_instances=0,
                max_instances=2,
            )
        self.ml_client.compute.begin_create_or_update(compute_cluster).result()

    def create_environment(self, env_name: str, env_conda_file):
        self.env = Environment(
            name=env_name,
            conda_file=env_conda_file,
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        )

    def use_existing_environment(self, env_name: str, version: any = "latest"):
        if version == "latest":
            self.env = f"{env_name}@latest"
        else:
            self.env = f"{env_name}:{version}"

    def create_deployment(self, scoring_script: str, compute_name: str):
        self.deployment = ModelBatchDeployment(
            name="iris-xgboost-depl",
            description="A deployment using xgboost to classify Iris data.",
            endpoint_name=self.endpoint_name,
            model=self.model,
            code_configuration=CodeConfiguration(
                code="assets/", scoring_script=scoring_script
            ),
            environment=self.env,
            compute=compute_name,
            settings=ModelBatchDeploymentSettings(
                max_concurrency_per_instance=1,
                mini_batch_size=10, # specific to this data set
                instance_count=1,
                output_action=BatchDeploymentOutputAction.APPEND_ROW,
                output_file_name="predictions.csv",
                retry_settings=BatchRetrySettings(max_retries=3, timeout=30),
                logging_level="info",
            ),
        )

        self.ml_client.begin_create_or_update(self.deployment).result()

    def set_deployment_as_default(self):    
        endpoint = self.ml_client.batch_endpoints.get(self.endpoint_name)
        endpoint.defaults.deployment_name = self.deployment.name
        self.ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    
    def invoke_batch_endpoint(self, output_location="default"):
        if output_location == "default":
            self.job = self.ml_client.batch_endpoints.invoke(
                endpoint_name=self.endpoint_name,
                deployment_name=self.deployment.name,
                input=Input(
                    path="assets/batch-requests",
                    type=AssetTypes.URI_FOLDER
                )
            )
        elif output_location == "azuredatastore":
            #  find the ID of a data store registered in AzureML
            self.batch_ds = self.ml_client.datastores.get_default()
            filename = f"predictions-{random.randint(0,99999)}.csv"

            self.job = self.ml_client.batch_endpoints.invoke(
                endpoint_name=self.endpoint_name,
                input=Input(
                    path="assets/batch-requests",
                    type=AssetTypes.URI_FOLDER,
                ),
                params_override=[
                    {"output_dataset.datastore_id": f"azureml:{self.batch_ds.id}"},
                    {"output_dataset.path": f"/{self.endpoint_name}/"},
                    {"output_file_name": filename},
                ],
            )

    def download_results(self, download_path):
        time.sleep(120) # ensuring all resources are available 
        scoring_job = list(self.ml_client.jobs.list(parent_job_name=
                                                    self.job.name))[0]
        self.ml_client.jobs.download(name=scoring_job.name, 
                                     download_path=download_path, 
                                     output_name="score")
        

if __name__ == '__main__':

    deployment = AzureMLBatchDeployment()
    # deployment.create_batch_endpoint(model_name="xgboost")
    deployment.use_existing_batch_endpoint(endpoint_name="xgboost-batch-8i8og")
    deployment.use_preregistered_model(model_name="xgboost_model")
    deployment.use_existing_environment(env_name="xgboost-batch-inference-env")
    # deployment.create_environment(env_name="xgboost-batch-inference-env", 
    #                               env_conda_file="assets/conda.yaml")
    deployment.create_deployment(scoring_script="batch_driver.py", 
                                 compute_name="ben-small-test")
    deployment.set_deployment_as_default()
    deployment.invoke_batch_endpoint()
    deployment.download_results(download_path=".")