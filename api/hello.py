from flask import Flask
from flask import request
import numpy as np 
from joblib import load

app = Flask(__name__)

best_model_path = "/home/r/hand_written_digit_recognition/hand_written_digit_recognition/models/test_0.15_val_0.15_rescale_8_gamma_0.001/model.joblib"
clf = load(best_model_path)
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict", methods = ['POST'])
def predict():
    input_jason = request.json
    image = input_jason['image']
    #print(image)
    image = np.array(image).reshape(1,-1)
    predicted = clf.predict(image)
    #return None
    return str(predicted[0])