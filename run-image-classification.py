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
run = experiment.submit(config)

# Track the run with MLflow
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_id=run.id):
    # Log parameters
    mlflow.log_param('compute_target', 'ben-small-test')
    mlflow.log_param('environment', env.name)

    # Monitor the run
    run.wait_for_completion(show_output=True)

    # Log metrics
    metrics = run.get_metrics()
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    # Log artifacts
    artifacts = run.get_file_names()
    for artifact in artifacts:
        if artifact.endswith('.pkl') or artifact.endswith('.txt'):
            mlflow.log_artifact(run.download_file(artifact))
