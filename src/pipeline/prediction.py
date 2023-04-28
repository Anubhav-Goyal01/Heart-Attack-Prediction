import os, sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictionPipeline:
    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

            model = load_object(file_path=model_path)
            preprocessor=load_object(file_path = preprocessor_path)

            data_preprocessed = preprocessor.transform(features)

            preds = model.predict_proba(data_preprocessed)
            preds_class = model.predict(data_preprocessed)
            print(preds_class)
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:

    def __init__(
      self,
      age:int,
      sex: str,
      cp: str,
      trtbps:int,
      oldpeak: float,
      chol: int,
      fbs: str,
      restecg: str,
      thall: str,
      slp: int,
      thalachh:int,
      exng:str,
      caa:str,   
    ) -> None:
        
        self.age = age
        self.sex = sex
        self.trtbps = trtbps
        self.cp = cp
        self.oldpeak = oldpeak
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thall = thall
        self.slp = slp
        self.thalachh = thalachh
        self.exng = exng
        self.caa = caa

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "cp": [self.cp],
                "trtbps": [self.trtbps],
                "chol": [self.chol],
                "fbs": [self.fbs],
                "restecg": [self.restecg],
                "thalachh": [self.thalachh],
                "exng": [self.exng],
                "oldpeak": [self.oldpeak],
                "slp": [self.slp],
                'caa': [self.caa],
                'thall': [self.thall],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)