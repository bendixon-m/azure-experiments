import mlflow
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.run import Run

# Load the Azure ML workspace
ws = Workspace.from_config()

# set up MLflow to track the metrics
experiment_name = "azure-image-classification"

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment(experiment_name)
mlflow.autolog()


# Set up the Azure ML experiment
experiment = Experiment(workspace=ws, name=experiment_name)
config = ScriptRunConfig(source_directory='./src',
                            script='train-image-classification.py',
                            compute_target='ben-small-test')

env = Environment.from_conda_specification(name='image-classification',
                                            file_path='azureml_envs/image-classification.yaml')
config.run_config.environment = env


# Submit the run
with mlflow.start_run() as run:
    experiment.submit(config)

# register the model
model_uri = "runs:/{}/model".format(run.info.run_id)
model = mlflow.register_model(model_uri, "sklearn_mnist_model")

print(model_uri)