import random
import string


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



class AzureMLDeployment:

    def __init__(self, compute_name: str, scoring_script_path: str):    
        self.credential = DefaultAzureCredential()
        self.ml_client = MLClient.from_config(self.credential)
        self.compute_name = compute_name
        self.compute_target = self.get_or_create_compute_target()
        self.scoring_script_path = scoring_script_path

    def create_endpoint(self):
        endpoint_name = "xgboost-batch"
        allowed_chars = string.ascii_lowercase + string.digits
        endpoint_suffix = "".join(random.choice(allowed_chars) for x in range(5))
        self.endpoint_name = f"{endpoint_name}-{endpoint_suffix}"
        self.endpoint = BatchEndpoint(
            name=endpoint_name,
            description="A batch endpoint for returning xgboost inferences"
        )
        self.ml_client.begin_create_or_update(self.endpoint).result()
        
    def register_model(self, model_name: str, model_local_path: str):
        xgboost_model = Model(
            path=model_local_path,
            type=AssetTypes.CUSTOM_MODEL,
            name=model_name,
            description="Model created from local files.",
        )
        self.ml_client.models.create_or_update(xgboost_model)

        self.model = #?
    
    def create_environment(self, env_name: str, packages: list):
        self.env = Environment(
            name="xgboost-batch-inference-env",
            conda_file="assets/conda.yaml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        )
        

    def create_deployment(self, compute: str):

        deployment = ModelBatchDeployment(
            name="iris-xgboost-depl",
            description="A deployment using xgboost to classify Iris data.",
            endpoint_name=self.endpoint_name,
            model=self.model,
            code_configuration=CodeConfiguration(
                code="assets/", scoring_script="batch_driver.py"
            ),
            environment=self.env,
            compute='ben-small-test',
            settings=ModelBatchDeploymentSettings(
                max_concurrency_per_instance=1,
                mini_batch_size=4,
                instance_count=1,
                output_action=BatchDeploymentOutputAction.APPEND_ROW,
                output_file_name="predictions.csv",
                retry_settings=BatchRetrySettings(max_retries=3, timeout=30),
                logging_level="info",
            ),
        )

        self.ml_client.begin_create_or_update(deployment).result()

        self.deployment_name = deployment.name

    



if __name__ == '__main__':

    deployment = AzureMLDeployment()
    deployment.create_endpoint()
    deployment.register_model(model_name = "xgboost_model", 
                   model_local_path = "assets/xgboost_model.json"
                )
    deployment.create_environment(env_name="xgboost-batch-inference-env",
                                  )
