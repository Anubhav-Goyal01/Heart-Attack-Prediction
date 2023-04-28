import os, sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import  SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")




class LogScaling(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self   

    def transform(self, X):
        return np.log1p(X)
    



class DataTransformation:

    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):

        try:
            log_scaling_columns = ['oldpeak', 'chol']
            cat_columns = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']
            num_columns = ['age', 'trtbps', 'thalachh']


            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(handle_unknown= 'ignore')),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer([
                ("log_transform", LogScaling(), log_scaling_columns),
                ("num_pipeline", num_pipeline, num_columns),
                ("cat_pipelines",cat_pipeline, cat_columns)
                ], remainder= 'passthrough')

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Data Read successfully")
            print(train_df)

            logging.info("Obtaining preprocessing object")
            preprocess_object = self.get_data_transformer_object()


            target_column_name = 'output'
            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]
            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]


            logging.info(f"Applying preprocessing object on training and test set")
            X_train_arr = preprocess_object.fit_transform(X_train)
            X_test_arr  = preprocess_object.transform(X_test)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocess_object
            )
            logging.info("Saved preprocessing object")


            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)