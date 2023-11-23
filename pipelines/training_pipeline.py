# from zenml.config import DockerSettings
# from zenml.integrations.constants import MLFLOW
from zenml import pipeline
from botocore.exceptions import EndpointProviderError

# docker_settings = DockerSettings(required_integrations=[MLFLOW])
from steps.clean_data import clean_df
from steps.evaluation import evaluation
from steps.ingest_data import ingest_df
from steps.model_train import train_model


# @pipeline(enable_cache=False, settings={"docker": docker_settings})
@pipeline(enable_cache=True)
def train_pipeline(data_path : str):
    """
   
    Returns:
        r2_score: float
        rmse: float
    """
    df = ingest_df(data_path)
    x_train, x_test, y_train, y_test = clean_df(df)
    model = train_model(x_train, x_test, y_train, y_test)
    r2_score, rmse = evaluation(model, x_test, y_test)
