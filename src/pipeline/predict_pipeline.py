import sys
from src.exception import CustomException
from src.utils import load_object
import pandas as pd


import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        Gender: int,
        Accommodation_Type: int,
        Do_you_work_while_studying: int,
        Participation_in_Clubs: int,
        Scholarship_Status: int,
        Academic_Support: int,
        Age_Range: int,
        Level_of_Study: int,
        Health_Challenges: int,
        School_Activities_Stress: int,
        Internet_Access: int,
        CGPA_Range: int,
        Parental_Level_of_Education: int,
        Hours_of_Study_per_Week: int,
        Class_Attendance: int,
        Faculty: str,
        Financial_Support_Source: str,
        Admission_Year: str
    ):

        self.Gender = Gender
        self.Accommodation_Type = Accommodation_Type
        self.Do_you_work_while_studying = Do_you_work_while_studying
        self.Participation_in_Clubs = Participation_in_Clubs
        self.Scholarship_Status = Scholarship_Status
        self.Academic_Support = Academic_Support
        self.Age_Range = Age_Range
        self.Level_of_Study = Level_of_Study
        self.Health_Challenges = Health_Challenges
        self.School_Activities_Stress = School_Activities_Stress
        self.Internet_Access = Internet_Access
        self.CGPA_Range = CGPA_Range
        self.Parental_Level_of_Education = Parental_Level_of_Education
        self.Hours_of_Study_per_Week = Hours_of_Study_per_Week
        self.Class_Attendance = Class_Attendance
        self.Faculty = Faculty
        self.Financial_Support_Source = Financial_Support_Source
        self.Admission_Year = Admission_Year


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.Gender],
                "Accommodation_Type": [self.Accommodation_Type],
                "Do_you_work_while_studying?": [self.Do_you_work_while_studying],
                "Participation_in_Clubs/Activities": [self.Participation_in_Clubs],
                "Scholarship_Status": [self.Scholarship_Status],
                "Do_you_receive_academic_support_(tutorials, mentorship, etc.)?": [self.Academic_Support],
                "Age_Range": [self.Age_Range],
                "Level_of_Study": [self.Level_of_Study],
                "Health_Challenges": [self.Health_Challenges],
                "School_Activities_Stress": [self.School_Activities_Stress],
                "Internet_Access": [self.Internet_Access],
                "CGPA_Range": [self.CGPA_Range],
                "Parental_Level_of_Education": [self.Parental_Level_of_Education],
                "Hours_of_Study_per_Week": [self.Hours_of_Study_per_Week],
                "Class_Attendance": [self.Class_Attendance],
                "Faculty": [self.Faculty],
                "Financial_Support_Source": [self.Financial_Support_Source],
                "Admission_Year": [self.Admission_Year]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        