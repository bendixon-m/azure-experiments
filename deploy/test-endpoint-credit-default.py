# Follows this tutorial https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-deploy-model?view=azureml-api-2

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential)

online_endpoint_name = 'credit-endpoint-dda6ef80'

print('Invoking endpoint')

# test the blue deployment with the sample data
response = ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name="blue",
    request_file="deploy/assets/sample-request.json",
)

print(response)

print('Getting logs')

logs = ml_client.online_deployments.get_logs(
    name="blue", endpoint_name=online_endpoint_name, lines=10
)
print(logs)