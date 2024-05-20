# Azure Experiments

This repo has some basic examples of training models using Azure. 

It was put together using these Azure docs:

* [Python get started](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-hello-world?view=azureml-api-1) with Azure ML, focused on running using the Azure ML portal
* Set up to run remotely, using [configure and submit training jobs](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets?view=azureml-api-1#select-a-compute-target)
* [Train and deploy image classification](https://github.com/Azure/MachineLearningNotebooks/tree/master/tutorials/compute-instance-quickstarts/quickstart-azureml-in-10mins) guide, which uses [these](https://github.com/Azure/MachineLearningNotebooks/tree/master/tutorials/compute-instance-quickstarts/quickstart-azureml-in-10mins) notebooks.

There are three jobs/runs defined here:
* `run-hello.py` - the "Hello world" run, a basic control script giving the URL for the run on the portal
* `run-scikit.py` - trains a model, uses a conda YAML file to configure environment requirements, e.g. numpy, scikit-learn
* `run-image-classification.py` - in progress!


### To run

To run any of these, we have to first authenticate with Azure, using `az login`. The workspace configuration file `config.json` was downloaded from the AZ portal. 