#predictions
import cv2
import numpy as np
# import torch  

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

def load_models(device):  #pass device as an argument
    #using OpenCV models
    age_net = cv2.dnn.readNetFromCaffe("C:/Users/elinz/OneDrive/CS/EPQ/models/age_deploy.prototxt", "C:/Users/elinz/OneDrive/CS/EPQ/models/age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe("C:/Users/elinz/OneDrive/CS/EPQ/models/gender_deploy.prototxt","C:/Users/elinz/OneDrive/CS/EPQ/models/gender_net.caffemodel" )
   # age_net = cv2.dnn.readNetFromCaffe("models/age_deploy.prototxt", "models/age_net.caffemodel")
    #gender_net = cv2.dnn.readNetFromCaffe("models/gender_deploy.prototxt", "models/gender_net.caffemodel")
    
    #ensure models are moved to the correct device if necessary (if using PyTorch instead of OpenCV)
    return age_net, gender_net

def predict_age_gender(face, age_net, gender_net):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104.0, 177.0, 123.0), swapRB=False)
    
    #gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    
    if gender_preds[0].max() > 0.6:  
        gender = gender_list[gender_preds[0].argmax()]
    else:
        gender = "Unknown"
        
    #age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]
    
    return gender, age
