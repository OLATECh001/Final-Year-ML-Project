import os
import sys
import pandas as pd
import numpy as np
import pickle

from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")

            logging.info("Loading trained model")
            model = pickle.load(open(model_path, "rb"))

            logging.info("Making prediction")

            # 🔥 XGBoost → NO SCALING
            data = features

            # 🔥 Use probability + threshold (VERY IMPORTANT)
            y_prob = model.predict_proba(data)[:, 1]
            y_pred = (y_prob > 0.2).astype(int)

            return y_pred

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_dataframe(self):
        try:
            return pd.DataFrame([self.data])
        except Exception as e:
            raise CustomException(e, sys)