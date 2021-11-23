from flask import Flask
from flask import request
import numpy as np 
from joblib import load
import os 
app = Flask(__name__)
#path = '/home/revanth/hand_written_digit_recognition'
#os.chdir(path)

best_model_path_svm = "../hand_written_digit_recognition/models/test_0.15_val_0.15_rescale_8_gamma_0.001/model.joblib"
clf_svm = load(best_model_path_svm)
best_model_path_dc = "../hand_written_digit_recognition/models/test_0.15_val_0.15_rescale_8_maxdepth_14/model.joblib"
clf_dc = load(best_model_path_dc)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/svm_predict", methods = ['POST'])
def svm_predict():
    input_jason = request.json
    image = input_jason['image']
    #print(image)
    image = np.array(image).reshape(1,-1)
    predicted = clf_svm.predict(image)
    #return None
    return str(predicted[0])

@app.route("/svm_decision_tree", methods = ['POST'])
def svm_decision_tree():
    input_jason = request.json
    image = input_jason['image']
    #print(image)
    image = np.array(image).reshape(1,-1)
    predicted = clf_dc.predict(image)
    #return None
    return str(predicted[0])