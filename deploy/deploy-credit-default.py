# Follows this tutorial https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-deploy-model?view=azureml-api-2

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# authenticate
credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient.from_config(credential)

# ### Register a model manually ###
# Run this later as this will be required for withdrawal model, for now use registered model `credit_defaults_model_ben_local`

# # Import the necessary libraries
# from azure.ai.ml.entities import Model
# from azure.ai.ml.constants import AssetTypes

# # Provide the model details, including the
# # path to the model files, if you've stored them locally.
# mlflow_model = Model(
#     path="./deploy/credit_defaults_model/",
#     type=AssetTypes.MLFLOW_MODEL,
#     name="credit_defaults_model",
#     description="MLflow Model created from local files.",
# )

# # Register the model
# ml_client.models.create_or_update(mlflow_model)

registered_model_name = "credit_defaults_model_ben_local"

# Let's pick the latest version of the model
latest_model_version = max(
    [int(m.version) for m in ml_client.models.list(name=registered_model_name)]
)

print(f"Latest model version: {latest_model_version}")

# ### Create endpoint ###

import uuid

# Create a unique name for the endpoint
online_endpoint_name = "credit-endpoint-" + str(uuid.uuid4())[:8]

from azure.ai.ml.entities import ManagedOnlineEndpoint

# define an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="this is an online endpoint and it's great!",
    auth_mode="key",
    tags={
        "training_dataset": "credit_defaults",
    },
)

# # create the online endpoint
# # expect the endpoint to take approximately 2 minutes.

endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)

print(
    f'Endpoint "{endpoint.name}" with provisioning state "{endpoint.provisioning_state}" is retrieved'
)

# create the online deployment


from azure.ai.ml.entities import ManagedOnlineDeployment



# Choose the latest version of the registered model for deployment
model = ml_client.models.get(name=registered_model_name, version=latest_model_version)

print(f'Fetched model: {model}')

# define an online deployment
# if you run into an out of quota error, change the instance_type to a comparable VM that is available.\
# Learn more on https://azure.microsoft.com/en-us/pricing/details/machine-learning/.
blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=online_endpoint_name,
    model=model,
    instance_type="Standard_DS3_v2",
    instance_count=1,
)

print('Create online deployment')
 
# create the online deployment
blue_deployment = ml_client.online_deployments.begin_create_or_update(
    blue_deployment
).result()

print('Create blue deployment')

# blue deployment takes 100% traffic
# expect the deployment to take approximately 8 to 10 minutes.
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print('Getting endpoint metadata')

# return an object that contains metadata for the endpoint
endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)

# print a selection of the endpoint's metadata
print(
    f"Name: {endpoint.name}\nStatus: {endpoint.provisioning_state}\nDescription: {endpoint.description}"
)

# existing traffic details
print(endpoint.traffic)

# Get the scoring URI
print(endpoint.scoring_uri)