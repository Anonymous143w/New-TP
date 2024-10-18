import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import numpy as np

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = LinearRegression()
        
    def train(self, X_train, y_train, X_test, y_test):
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        with mlflow.start_run():
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Log metrics
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            mlflow.log_metric("test_rmse", test_rmse)
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
            
            # Save model locally
            joblib.dump(self.model, self.config['model']['output_path'])
            
        return self.model