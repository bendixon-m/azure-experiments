from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# authenticate
credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient.from_config(credential)

### Register a model manually ###

# Import the necessary libraries
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

# Provide the model details, including the
# path to the model files, if you've stored them locally.
mlflow_model = Model(
    path="./assets/xgboost_model/",
    type=AssetTypes.MLFLOW_MODEL,
    name="credit_defaults_model",
    description="MLflow Model created from local files.",
)

# Register the model
ml_client.models.create_or_update(mlflow_model)