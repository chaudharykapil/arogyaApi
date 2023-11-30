# Importing necessary Libraries
from flask import Flask, request, redirect
import pickle 
import numpy as np
import json
app = Flask(__name__)
app.app_context().push()


#Loading Diease Dectection Pickle File
f = open("DecisionTree-Model.sav", "rb")
model_N = pickle.load(f)

#loading Medicine Recommendation Pickle file
f2 = open("drugTree.pkl", "rb")
model_med = pickle.load(f2)


img_labels = pickle.load(open("labels.pkl",'rb'))

symptom_mapping = {
    'acidity': 0,
    'indigestion': 1,
    'headache': 2,
    'blurred_and_distorted_vision': 3,
    'excessive_hunger': 4,
    'muscle_weakness': 5,
    'stiff_neck': 6,
    'swelling_joints': 7,
    'movement_stiffness': 8,
    'depression': 9,
    'irritability': 10,
    'visual_disturbances': 11,
    'painful_walking': 12,
    'abdominal_pain': 13,
    'nausea': 14,
    'vomiting': 15,
    'blood_in_mucus': 16,
    'Fatigue': 17,
    'Fever': 18,
    'Dehydration': 19,
    'loss_of_appetite': 20,
    'cramping': 21,
    'blood_in_stool': 22,
    'gnawing': 23,
    'upper_abdomain_pain': 24,
    'fullness_feeling': 25,
    'hiccups': 26,
    'abdominal_bloating': 27,
    'heartburn': 28,
    'belching': 29,
    'burning_ache': 30
}




def medicineValidation(selectedOptions):
    """Defining a function to recommend medicine"""
    print("input",selectedOptions)
    inputs = np.array(selectedOptions)
    inputs = inputs.reshape(1, -1)
    recommend_Med = model_med.predict(inputs)
    return recommend_Med[0]

def serviceValidation(selected_symptoms):

    # Convert the selected symptoms to a 30-element list of 1s and 0s

    inputs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for symptom in selected_symptoms:
        if symptom:
            inputs[symptom_mapping[symptom]] = 1
    print("input",inputs)
    # convert list to NumPy array
    inputs = np.array(inputs)
    inputs = inputs.reshape(1, -1)


    # Pass the inputs to your machine learning model and retrieve the predicted result
    predicted_result = model_N.predict(inputs)
    print(predicted_result[0])

    # Return the predicted result to the user
    return predicted_result[0]




@app.route("/test")
def test():
    print()
    return "ok"


@app.route("/test-medicine",methods = ["POST"])
def MedicinePred():
    data = json.loads(request.data)
    print(data)
    selectedOptions = data
    out = medicineValidation(selectedOptions)
    print(out)
    return json.dumps({"res":out})


@app.route("/test-disease", methods = ["POST"])
def DiseasePred():
    data = request.data
    data = json.loads(data)
    print(data)
    return json.dumps({"res":0})


# if __name__ == "__main__":
#     app.run(debug=True)