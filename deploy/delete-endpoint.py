from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# authenticate
credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient.from_config(credential)

online_endpoint_name = 'credit-endpoint-a12fd767'
ml_client.online_endpoints.begin_delete(name=online_endpoint_name).result()