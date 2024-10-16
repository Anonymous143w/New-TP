import pandas as pd
from sklearn.model_selection import train_test_split
import logging

class DataIngestion:
    def __init__(self, config):
        self.config = config
        
    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion")
        try:
            df = pd.read_csv(self.config['data']['raw_data_path'])
            
            train_set, test_set = train_test_split(
                df,
                test_size=self.config['data']['test_size'],
                random_state=self.config['data']['random_state']
            )
            
            return train_set, test_set
            
        except Exception as e:
            logging.error(f"Exception occurred in data ingestion: {e}")
            raise e