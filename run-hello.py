# get-started/run-hello.py
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig

ws = Workspace.from_config()
experiment = Experiment(workspace=ws, name='day1-experiment-hello-local')

config = ScriptRunConfig(source_directory='./src', script='hello.py', compute_target='ben-small-test')

run = experiment.submit(config)
aml_url = run.get_portal_url()
print(aml_url)