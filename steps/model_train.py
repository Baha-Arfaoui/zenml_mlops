import logging 

from botocore.exceptions import EndpointProviderError

from zenml import step 
import pandas as pd 
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
from src.model_dev import LinearRegressionModel
@step
def train_model(x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig) -> RegressorMixin:
    """ 
    Trains the model on the ingested data .
     
    Args:
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    """
    
    model = None
    if config.model_name == "LinearRegression":
        model= LinearRegressionModel()
        trained_model=model.train(x_train,y_train)
        return train_model
    else: 
        raise ValueError(f"Model  {config.model_name}  not supported")

