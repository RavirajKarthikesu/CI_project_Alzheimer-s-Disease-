from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

#  Define Flask app
app = Flask(__name__)

#  Load trained model
try:
    with open("rf_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    print(" Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")

#  Define expected fields
expected_fields = [
    "PatientID", "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking",
    "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality",
    "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes", "Depression",
    "HeadInjury", "Hypertension", "SystolicBP", "DiastolicBP", "CholesterolTotal",
    "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", "MMSE",
    "FunctionalAssessment", "MemoryComplaints", "BehavioralProblems", "ADL",
    "Confusion", "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
    "Forgetfulness"
]

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        print(" Received POST request")
        print(" Form Data:", request.form)  # Debugging print

        try:
            #  Ensure all fields exist in the request
            missing_fields = [field for field in expected_fields if field not in request.form]
            if missing_fields:
                print(f" Missing fields: {missing_fields}")
                return render_template('index.html', prediction=" Error: Missing form data.")

            #  Extract & Convert Data
            form_data = {key: float(request.form[key]) for key in expected_fields}
            
            #  Convert to NumPy Array
            features = np.array([list(form_data.values())])

            #  Validate Feature Count
            if features.shape[1] != model.n_features_in_:
                print(f" Error: Model expects {model.n_features_in_} features, but got {features.shape[1]}")
                return render_template('index.html', prediction=" Error: Incorrect feature count.")

            #  Make Prediction
            prediction = model.predict(features)[0]
            print(" Prediction:", prediction)
            

        except Exception as e:
            print(f" Error during processing: {e}")
            return render_template('index.html', prediction=" Error: Could not process data.")

    return render_template('index.html', prediction=prediction)

#  Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
