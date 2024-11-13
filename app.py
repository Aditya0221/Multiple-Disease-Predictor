from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn import impute
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np



# Initialize the Flask app
app = Flask(__name__)


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes', methods=['POST', 'GET'])
def diabetes(): # Get the form data from the HTML form
    if request.method == 'POST':
        pregnancies = request.form.get('pregnancies')
        glucose = request.form.get('glucose')
        bloodpressure = request.form.get('bloodpressure')
        skinthickness = request.form.get('skinthickness')
        insulin = request.form.get('insulin')
        bmi = request.form.get('bmi') #bmi = request.form['bmi']
        dpf = request.form.get('dpf') #dpf = request.form['dpf']
        age = request.form.get('age') #age = request.form['age']
        model = joblib.load('diabetes_model.sav')
        values = np.asarray([pregnancies,glucose,bloodpressure,skinthickness,insulin,bmi,dpf,age])  
        pred = model.predict(values.reshape(1, -1))[0]
        tracker = 1
        return render_template('diabetes.html', pred = pred, tracker = tracker)
    tracker = 0     
        
    return render_template('diabetes.html', tracker = tracker)


@app.route('/heartdisease', methods=['POST', 'GET'])
def heartdisease(): # Get the form data from the HTML form
    if request.method == 'POST':
        age = float(request.form.get('age'))
        sex = float(request.form.get('sex'))
        cp = float(request.form.get('cp'))
        trestbps = float(request.form.get('trestbps'))
        chol = float(request.form.get('chol'))
        fbs = float(request.form.get('fbs'))
        restecg = float(request.form.get('restecg'))
        thalach = float(request.form.get('thalach'))
        exang = float(request.form.get('exang'))
        oldpeak = float(request.form.get('oldpeak'))
        slope = float(request.form.get('slope'))
        ca = float(request.form.get('ca'))
        thal = float(request.form.get('thal'))
        model = joblib.load('heart_disease_model.sav')
        values = np.asarray([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])  
        pred = model.predict(values.reshape(1, -1))[0]
        tracker = 1
        return render_template('heart.html', pred = pred, tracker = tracker)
    tracker = 0
    return render_template('heart.html', tracker = 0)


@app.route('/breastcancer', methods=['POST', 'GET'])
def breastcancer(): # Get the form data from the HTML form
    if request.method == 'POST':
            radius_mean = float(request.form.get('radius_mean'))
            texture_mean = float(request.form.get('texture_mean'))
            perimeter_mean = float(request.form.get('perimeter_mean'))
            area_mean = float(request.form.get('area_mean'))
            smoothness_mean = float(request.form.get('smoothness_mean'))
            compactness_mean = float(request.form.get('compactness_mean'))
            concavity_mean = float(request.form.get('concavity_mean'))
            concave_points_mean = float(request.form.get('concave_points_mean'))
            symmetry_mean = float(request.form.get('symmetry_mean'))
            fractal_dimension_mean = float(request.form.get('fractal_dimension_mean'))
            radius_se = float(request.form.get('radius_se'))
            texture_se = float(request.form.get('texture_se'))
            perimeter_se = float(request.form.get('perimeter_se'))
            area_se = float(request.form.get('area_se'))
            smoothness_se = float(request.form.get('smoothness_se'))
            compactness_se = float(request.form.get('compactness_se'))
            concavity_se = float(request.form.get('concavity_se'))
            concave_points_se = float(request.form.get('concave_points_se'))
            symmetry_se = float(request.form.get('symmetry_se'))
            fractal_dimension_se = float(request.form.get('fractal_dimension_se'))
            
            radius_worst = float(request.form.get('radius_worst'))
            texture_worst = float(request.form.get('texture_worst'))
            perimeter_worst = float(request.form.get('perimeter_worst'))
            area_worst = float(request.form.get('area_worst'))
            smoothness_worst = float(request.form.get('smoothness_worst'))
            compactness_worst = float(request.form.get('compactness_worst'))
            concavity_worst = float(request.form.get('concavity_worst'))
            concave_points_worst = float(request.form.get('concave_points_worst'))
            symmetry_worst = float(request.form.get('symmetry_worst'))
            fractal_dimension_worst = float(request.form.get('fractal_dimension_worst'))
            
            # Create feature array for prediction
            features = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean,
                                  smoothness_mean, compactness_mean, concavity_mean,
                                  concave_points_mean, symmetry_mean, fractal_dimension_mean,
                                  radius_se, perimeter_se, area_se, compactness_se,
                                  concavity_se, concave_points_se, fractal_dimension_se,
                                  radius_worst, texture_worst, perimeter_worst, area_worst,
                                  smoothness_worst, compactness_worst, concavity_worst,
                                  concave_points_worst, symmetry_worst, fractal_dimension_worst, smoothness_se, symmetry_se, texture_se]])
            model = joblib.load('breast-cancer2.sav')  
            pred = model.predict(features.reshape(1, -1))[0]
            tracker = 1
            return render_template('breast_cancer.html', pred = pred, tracker = tracker)
    tracker = 0
    return render_template('breast_cancer.html', tracker = 0)

@app.route('/lung-cancer', methods=['GET', 'POST'])
def lungcancer():
    if request.method == 'POST':
            # Convert form inputs to the correct numeric types
            yellow_fingers = int(request.form.get('yellow_fingers'))
            anxiety = int(request.form.get('anxiety'))
            peer_pressure = int(request.form.get('peer_pressure'))
            chronic_disease = int(request.form.get('chronic_disease'))
            fatigue = int(request.form.get('fatigue'))
            allergy = int(request.form.get('allergy'))
            wheezing = int(request.form.get('wheezing'))
            alcohol_consuming = int(request.form.get('alcohol_consuming'))
            coughing = int(request.form.get('coughing'))
            shortness_of_breath = int(request.form.get('shortness_of_breath'))
            swallowing_difficulty = int(request.form.get('swallowing_difficulty'))
            chest_pain = int(request.form.get('chest_pain'))

            # Load the model
            model = joblib.load('lung-cancer-prediction2.sav')
            
            # Prepare the data for prediction
            values = np.asarray([
                yellow_fingers, anxiety, peer_pressure, 
                chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, 
                coughing, shortness_of_breath, swallowing_difficulty, chest_pain
            ])
            
            # Make prediction
            pred = model.predict(values.reshape(1, -1))[0]
            tracker = 1
            
            return render_template('lungcancer.html', pred=pred, tracker=tracker)
    
    
    return render_template('lungcancer.html', tracker=0)

@app.route('/liver', methods=['GET', 'POST'])
def liverdisease():
    if request.method == 'POST':
            # Convert form inputs to the correct numeric types
            age = float(request.form.get('Age'))
            gender = int(request.form.get('Gender'))  # Assuming 1 for male and 0 for female
            bmi = float(request.form.get('BMI'))
            alcohol_consumption = float(request.form.get('AlcoholConsumption'))
            smoking = int(request.form.get('Smoking'))  # Assuming 1 for yes and 0 for no
            genetic_risk = int(request.form.get('GeneticRisk'))  # Assuming 1 for yes and 0 for no
            physical_activity = float(request.form.get('PhysicalActivity'))  # Hours per week
            diabetes = int(request.form.get('Diabetes'))  # Assuming 1 for yes and 0 for no
            hypertension = int(request.form.get('Hypertension'))  # Assuming 1 for yes and 0 for no
            liver_function_test = float(request.form.get('LiverFunctionTest'))

            scaler = StandardScaler()
            # Load the model
            model = joblib.load('liver-disease2.sav')
            
            # Prepare the data for prediction
            values = np.asarray([
                age, gender, bmi, alcohol_consumption, smoking, genetic_risk, physical_activity, diabetes, hypertension, liver_function_test])
            values = scaler.fit_transform(values.reshape(-1, 1))
            
            # Make prediction
            pred = model.predict(values.reshape(1, -1))[0]
            tracker = 1
            
            return render_template('liver.html', pred=pred, tracker=tracker)
    
    
    return render_template('liver.html', tracker=0)


@app.route('/consult')
def consult():
    return render_template('forms.html')
 
if __name__ == '__main__':
    app.run(debug=True)