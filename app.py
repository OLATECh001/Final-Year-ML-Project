from flask import Flask,request,render_template
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application


# Route for a home page
@app.route('/')
def home_page():
    return render_template('index.html')

# Prediction Route
@app.route('/predictstudentdata', methods=['GET', 'POST'])
def predict_datapoint():
    print(request.form)
    from src.pipeline.predict_pipeline import CustomData,PredictPipeline

    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        data = CustomData(
            Gender=int(request.form.get('Gender')),
            Accommodation_Type=int(request.form.get('Accommodation_Type')),
            Do_you_work_while_studying=int(request.form.get('Do_you_work_while_studying?')),
            Participation_in_Clubs=int(request.form.get('Participation_in_Clubs/Activities')),
            Scholarship_Status=int(request.form.get('Scholarship_Status')),
            Academic_Support=int(request.form.get('Do_you_receive_academic_support_(tutorials, mentorship, etc.)?')),

            Age_Range=int(request.form.get('Age_Range')),
            Level_of_Study=int(request.form.get('Level_of_Study')),
            Health_Challenges=int(request.form.get('Health_Challenges')),
            School_Activities_Stress=int(request.form.get('School_Activities_Stress')),
            Internet_Access=int(request.form.get('Internet_Access')),
            CGPA_Range=int(request.form.get('CGPA_Range')),
            Parental_Level_of_Education=int(request.form.get('Parental_Level_of_Education')),
            Hours_of_Study_per_Week=int(request.form.get('Hours_of_Study_per_Week')),
            Class_Attendance=int(request.form.get('Class_Attendance')),

            Faculty=request.form.get('Faculty'),
            Financial_Support_Source=request.form.get('Financial_Support_Source'),
            Admission_Year=request.form.get('Admission_Year')
        )
                
        pred_df = data.get_data_as_data_frame()
        print("DATAFRAME:\n", pred_df)

        predict_pipeline = PredictPipeline()
        # results = predict_pipeline.predict(pred_df)

        # print("PREDICTION RESULTS:", results)

        # predict_pipeline = PredictPipeline()

        # Prediction
        results = predict_pipeline.predict(pred_df)

        # 🔥 Probability (confidence score)
        proba = predict_pipeline.model.predict_proba(
            predict_pipeline.preprocessor.transform(pred_df)
        )[0][1]
    
        confidence = round(proba * 100, 2)

        print("PREDICTION RESULTS:", results)
        print("CONFIDENCE:", confidence)

        # Label
        if results[0] == 1:
            prediction_text = f"High Risk of Dropout ({confidence}% confidence)"
        else:
            prediction_text = f"Low Risk of Dropout ({confidence}% confidence)"

        return render_template('home.html', results=results[0], prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)