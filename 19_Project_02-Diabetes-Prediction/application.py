import pickle
from flask import Flask, render_template, request,jsonify
import numpy as np
import pandas as pd    
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

# Importing logistic regression and standard scaler pickle.
log_reg = pickle.load(open("./Models/logisticRegression.pkl", "rb"))    
standard_scaler = pickle.load(open("./Models/standardScaler.pkl", "rb"))   

# Routing for homepage

@application.route('/', methods=['GET','POST']) 
def predict_class():
    result = ""
    if request.method == 'POST':
        try:
            Pregnancies = int(request.form.get('Pregnancies'))            
            Glucose = float(request.form.get('Glucose'))
            BloodPressure = float(request.form.get('BloodPressure'))
            SkinThickness = float(request.form.get('SkinThickness'))
            Insulin = float(request.form.get('Insulin')) 
            BMI = float(request.form.get('BMI'))
            DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
            Age =  float(request.form.get('Age'))

            
            # Scaling
            new_scaled_data = standard_scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        
            # Prediction
            predict = log_reg.predict(new_scaled_data)
            
        
            if predict[0] == 0:
                result = "You don't have Diabetes"
            else:
                result = "You have Diabetes"    
        except (ValueError, TypeError) as e:
            result = "Invalid input. Please enter numeric value."  
               
            
        return render_template('result.html',result = result)
    else:
        return render_template('home.html')
    
if __name__ == "__main__":
    application.run(debug=True, host = "0.0.0.0", port=5000)    
