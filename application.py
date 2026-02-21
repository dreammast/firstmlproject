from flask import Flask,request,render_template,jsonify
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__, template_folder='templetes')

app = application

# route the appplication

@app.route('/',methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html')
    else:
        try:
            # Handle JSON requests from JavaScript fetch API
            if request.is_json:
                request_data = request.get_json()
            else:
                # Handle form data from regular form submission
                request_data = request.form
            
            data = CustomData(
                gender = request_data.get('gender'),
                race_ethnicity = request_data.get('race_ethnicity'),
                parental_level_of_education = request_data.get('parental_level_of_education'),
                lunch = request_data.get('lunch'),
                test_preparation_course = request_data.get('test_preparation_course'),
                reading_score = float(request_data.get('reading_score')),
                writing_score = float(request_data.get('writing_score'))
            )
            pred_df = data.get_data_as_dataframe()
            print(pred_df)

            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(pred_df)
            
            # If JSON request, return JSON response
            if request.is_json:
                return jsonify({'math_score': float(pred[0])})
            else:
                # If form request, return rendered template
                return render_template('home.html', prediction_text=f"The predicted math score is {pred[0]}")
        
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            if request.is_json:
                return jsonify({'error': str(e)}), 400
            else:
                return render_template('home.html', error_text=f"Error: {str(e)}")
    
if __name__ == "__main__":
    app.run(host="0.0.0.0")

