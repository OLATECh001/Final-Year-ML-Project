import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This is responsible for data transformation
        '''

        try:
            numerical_columns = [
                "CGPA_Range",
                "Parental_Level_of_Education",
                "Hours_of_Study_per_Week",
                "Class_Attendance",
                "Age_Range",
                "Level_of_Study",
                "Health_Challenges",
                "School_Activities_Stress",
                "Internet_Access",

                # binary features
                "Gender",
                "Accommodation_Type",
                "Do_you_work_while_studying?",
                "Participation_in_Clubs/Activities",
                "Scholarship_Status",
                "Do_you_receive_academic_support_(tutorials, mentorship, etc.)?"
            ]
           
            categorical_columns = [
                "Admission_Year",
                "Faculty",
                "Financial_Support_Source",
            ]
            

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ]   
            )


            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor

        except Exception as e:
            logging.info("Error in data transformation")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "Dropout_Intention?"
           

            input_feature_train_data = train_data.drop(columns=[target_column_name])
            target_feature_train_data = train_data[target_column_name]

            input_feature_test_data = test_data.drop(columns=[target_column_name])
            target_feature_test_data = test_data[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_data)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_data)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_data)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Error in initiate data transformation")
            raise CustomException(e, sys)
