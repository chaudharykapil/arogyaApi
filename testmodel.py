from utils.support import ConvolutionalNetwork,transform
import pickle
import torch
from PIL import Image
f = open("classes.pkl","rb") 
class_name = pickle.load(f)
f.close()
model_1 = ConvolutionalNetwork(class_name)
model_1.load_state_dict(torch.load("input_embeddings.pt"))
load_img = Image.open("static/input/img.jpg")
t_img = transform(load_img) 
print(class_name[model_1(t_img).argmax(dim=1).item()])