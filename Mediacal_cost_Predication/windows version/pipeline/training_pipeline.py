from src.data.data_ingestion import DataIngestion
from src.features.preprocessor import DataPreprocessor
from src.models.model_trainer import ModelTrainer
import yaml
import logging

class TrainingPipeline:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
    def run_pipeline(self):
        try:
            # Data Ingestion
            data_ingestion = DataIngestion(self.config)
            train_data, test_data = data_ingestion.initiate_data_ingestion()
            
            # Data Preprocessing
            preprocessor = DataPreprocessor(self.config)
            X_train, y_train = preprocessor.transform_features(train_data)
            X_test, y_test = preprocessor.transform_features(test_data)
            
            # Model Training
            trainer = ModelTrainer(self.config)
            model = trainer.train(X_train, y_train, X_test, y_test)
            
            return model
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise e