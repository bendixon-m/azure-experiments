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
xgboost_model = Model(
    path="./deploy/assets/xgboost_model/",
    type=AssetTypes.CUSTOM_MODEL,
    name="xgboost_model",
    description="Model created from local files.",
)

# Register the model
ml_client.models.create_or_update(xgboost_model)

registered_model_name = "xgboost_model"

# Let's pick the latest version of the model
latest_model_version = max(
    [int(m.version) for m in ml_client.models.list(name=registered_model_name)]
)

print(latest_model_version)