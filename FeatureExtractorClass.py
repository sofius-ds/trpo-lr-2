import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data["hour"] = data["timestamp"].dt.hour
            data["dayofweek"] = data["timestamp"].dt.weekday
            data = data.drop(columns="timestamp")

        return data