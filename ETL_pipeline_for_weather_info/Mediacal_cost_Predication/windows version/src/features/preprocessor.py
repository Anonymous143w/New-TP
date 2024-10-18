from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.le = LabelEncoder()
        
    def transform_features(self, df):
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['sex', 'smoker', 'region']
        for col in categorical_cols:
            df_processed[col] = self.le.fit_transform(df_processed[col])
        
        # Split features and target
        X = df_processed.drop(['charges'], axis=1)
        y = df_processed['charges']
        
        # Apply polynomial features if specified
        if self.config['model']['type'] == 'polynomial':
            poly = PolynomialFeatures(degree=self.config['model']['degree'])
            X = poly.fit_transform(X)
        
        return X, y
