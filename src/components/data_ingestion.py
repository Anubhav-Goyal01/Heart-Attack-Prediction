import os, sys
from src.logger import logging
from src.exception import CustomException


import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")


class DataIngestion:

    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    
    def initiate_data_ingestion(self):
        try:
            logging.info("Running data ingestion component")
            data = pd.read_csv('Data/heart.csv')
            logging.info("Dataset read successfully")
            

            split = StratifiedShuffleSplit(n_splits= 1, test_size= 0.2, random_state=42)

            for train_idx, test_idx in split.split(data, data['caa']):
                strat_train_set = data.loc[train_idx]
                strat_test_set = data.loc[test_idx]

            logging.info("Saving train and test datasets")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok= True)
            strat_train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)

            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok= True)
            strat_test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)


            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        



