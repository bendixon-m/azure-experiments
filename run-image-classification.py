import mlflow
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.run import Run

# Load the Azure ML workspace
ws = Workspace.from_config()

# set up MLflow to track the metrics
experiment_name = "azure-ml-in10-mins-tutorial"
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment(experiment_name)
mlflow.autolog()

# Start logging to a new run in the experiment
with mlflow.start_run() as run:
    
    # Set up the Azure ML experiment
    # experiment = Experiment(workspace=ws, name='day2-image-classification')
    config = ScriptRunConfig(source_directory='./src',
                             script='train-image-classification.py',
                             compute_target='ben-small-test')

    env = Environment.from_conda_specification(name='image-classification',
                                               file_path='image-classification.yaml')
    config.run_config.environment = env
    run = mlflow.start_run()

    # azure_run = experiment.submit(config)

    # Log the Azure ML run URL to MLflow
    mlflow.log_param("azure_run_url", azure_run.get_portal_url())

    # Wait for the run to complete
    azure_run.wait_for_completion(show_output=True)

    # Log the run results
    for metric_name, metric_value in azure_run.get_metrics().items():
        mlflow.log_metric(metric_name, metric_value)

    # Register the model with MLflow
    mlflow.azureml.log_model(azure_model=azure_run, registered_model_name="image-classification-model")
