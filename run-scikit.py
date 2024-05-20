# run-scikit.py

from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig

if __name__ == "__main__":
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='day1-experiment-train-scitkit-local')
    config = ScriptRunConfig(source_directory='./src', script='train.py', compute_target='ben-small-test')

    # set up environment
    env = Environment.from_conda_specification(name='scikit-env', file_path='scikit-env.yaml')

    config.run_config.environment = env

    run = experiment.submit(config)

    aml_url = run.get_portal_url()
    print(aml_url)