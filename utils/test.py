import pickle
f = open("F:\Projects\Proj_1\AYUCARE\Samudini\DecisionTree-Model.sav", "rb")
model_N = pickle.load(f)

#loading Medicine Recommendation Pickle file
f2 = open("F:\Projects\Proj_1\AYUCARE\Samudini\drugTree.pkl", "rb")
img_labels = pickle.load(open("F:\Projects\Proj_1\AYUCARE\Samudini\labels.pkl",'rb'))
model_med = pickle.load(f2)
from joblib import dump, load
# dump(model_N, 'DecisionTree-Model2.joblib')
dump(img_labels, 'labels.joblib') 
# dump(model_med, 'drugTree2.joblib') 