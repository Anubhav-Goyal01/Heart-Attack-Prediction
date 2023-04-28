import os, sys
from dataclasses import dataclass

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")



class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting train and test array")
            X_train, y_train, X_test, Y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                'xgboost' : XGBClassifier(),
                'catboost' : CatBoostClassifier(verbose=0),
                'lightgbm' : LGBMClassifier(),
                'gradient boosting' : GradientBoostingClassifier(),
                'random forest' : RandomForestClassifier(),
            }

            params={
                "xgboost":{
                    'eta':[.1,.01,.05,.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256, 512],
                    'max_depth': [6, 8, 10],
                },

                "catboost":{
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators' : [8, 16, 32, 64, 128, 256, 512],
                },

                "lightgbm":{
                    'learning_rate': [.1,.01,0.5,.001],
                    'max_depth': [6,8,10],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "gradient boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "random forest":{
                    'n_estimators': [8,16,32,64,128,256],
                    'max_depth': [6,8,10],
                }           
            }

            model_report:dict = evaluate_models(X_train, y_train, X_test, Y_test, models, params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best Model Found")

            logging.info("Model trained successfully")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path, 
                obj = best_model,
            )

            predicted = best_model.predict(X_test)
            accuracyScore = accuracy_score(Y_test, predicted)
            return accuracyScore


        except Exception as e:
            raise CustomException(e, sys)