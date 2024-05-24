# Azure Experiments

This repo has some basic examples of training models using Azure. 

It was put together using these Azure docs:

* [Python get started](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-hello-world?view=azureml-api-1) with Azure ML, focused on running using the Azure ML portal
* Set up to run remotely, using [configure and submit training jobs](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets?view=azureml-api-1#select-a-compute-target)
* [Train a model in Azure ML, SDK v2](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-train-model?view=azureml-api-2) which includes registering the model with mlflow

There are several command jobs defined here, which mostly run custom training jobs:
* `run-hello.py` - the "Hello world" run, a basic control script giving the URL for the run on the portal but executing no code
* `run-scikit.py` - trains a model, uses a conda YAML file to configure environment requirements, e.g. numpy, scikit-learn
* `run-credit-default.py` - trains a model, logs results using mlflow, store results

### Deployment 
Azure ML creates **endpoints** - a HTTPS path: a URI with TLS and some authentication method, and **deployments** - the resources for hosting the model.  

There is a file to deploy models:
* `deploy-credit-default.py` - creates an online endpoint using a model trained online
* `deploy-xgboost.py` - this loads a trained xgboost model saved as a json, and then creates an endpoint and deployment. Since it is a custom model, this script also [registers the environment](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&tabs=python#register-your-model-and-environment) required for the deployment.

## To run

To run any of these, we have to first authenticate with Azure, using `az login`. The workspace configuration file `config.json` was downloaded from the AZ portal. 

You also need to set up a compute instance and assign a Managed Identity to it, per [these](https://learn.microsoft.com/en-us/answers/questions/1377394/failed-to-pull-docker-image-from-acr-to-azure-ml) instructions. Once set up the compute resource needs to be in the `running` state to run a training job. 