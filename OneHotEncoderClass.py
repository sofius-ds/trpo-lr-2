import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class MyOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, target):
        self.target = target
        self.encoding = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.cat_cols = []

    def fit(self, X, y=None):
        self.cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        if self.target in self.cat_cols:
            self.cat_cols.remove(self.target)

        if self.cat_cols:
            self.encoding.fit(X[self.cat_cols])

        return self
    
    def transform(self, X):
        data = X.copy()

        target = data[self.target] if self.target in data.columns else None
        features = data.drop(columns=self.target) if self.target in data.columns else data

        if self.cat_cols:
            encoded_cols = self.encoding.transform(features[self.cat_cols])
            df_encoded = pd.DataFrame(
                encoded_cols,
                columns=self.encoding.get_feature_names(self.cat_cols),
                index=data.index
            )
            features = pd.concat([features, df_encoded], axis=1)
            features = features.drop(columns=self.cat_cols)

        return features, target
