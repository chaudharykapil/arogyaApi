# Importing necessary Libraries
from flask import Flask, request, redirect ,render_template
import pickle 
import numpy as np
import json
from keras.models import load_model
from utils.Residual import ResidualUnit,DefaultConv2D
from PIL import Image
from joblib import dump, load
import warnings
from sklearn.exceptions import DataConversionWarning

from utils.support import ConvolutionalNetwork,transform
import pickle
import torch




app = Flask(__name__)
app.app_context().push()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
#Loading Diease Dectection Pickle File
f = open("DecisionTree-Model.sav", "rb")
model_N = pickle.load(f)

#loading Medicine Recommendation Pickle file
f2 = open("drugTree.pkl", "rb")
model_med = pickle.load(f2)

# img_model = load_model("plant_model_v1.h5",custom_objects={"ResidualUnit":ResidualUnit,})
# img_labels = load('labels.joblib') 

f = open("classes.pkl","rb") 
img_class_name = pickle.load(f)
f.close()


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
    print("Disease model input: ",inputs)
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
    print(type(data))
    selectedOptions = data
    out = medicineValidation(selectedOptions)
    print(out)
    return json.dumps({"res":out})


@app.route("/test-disease", methods = ["POST"])
def DiseasePred():
    data = request.data
    data = json.loads(data)
    print("DiseaseData: ",data)
    out = serviceValidation(data)
    print(type(data))
    return json.dumps({"res":out})


# @app.route("/search-image/v1",methods = ["GET","POST"])
# def searchImage():
#     if request.method == "GET":
#         return "image api ok"
#     if request.method == "POST":
#         img = request.files.get("image")
#         img.save("static/input/img.jpg")
#         load_img = Image.open("static/input/img.jpg")
#         load_img = load_img.resize((128,128))
#         data = np.array(load_img).reshape(1,128,128,3)
#         print(data.shape)
#         res = img_model.predict(data)
#         prob  =  list(res[0])
#         idx = prob.index(max(prob))
#         print("Result: ",img_labels[idx])
#     return img_labels[idx]

@app.route("/search-image/v2",methods = ["GET","POST"])
def searchPlant():
    if request.method == "GET":
        return "image api V2 ok"
    img = request.files.get("image")
    img.save("static/input/img.jpg")
    load_img = Image.open("static/input/img.jpg")
    model_1 = ConvolutionalNetwork(img_class_name)
    model_1.load_state_dict(torch.load("input_embeddings.pt"))
    t_img = transform(load_img) 
    choice = model_1(t_img).argmax(dim=1).item()
    print(choice,img_class_name[choice])
    return img_class_name[choice]


if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0")