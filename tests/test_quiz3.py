import math
from sklearn import datasets, svm, metrics, tree
import os
from joblib import load, dump
import numpy as np
import sys  
path = os.getcwd()
#print(path)
path_2 = path[:-5] + 'hand_written_digit_recognition/'
os.chdir(path_2)
#print(os.getcwd())
#sys.path.append('./hand_written_digit_recognition/hand_written_digit_recognition') 
from utils import preprocess, create_splits, run_classification_experiment

best_model_path_svm = "./models/test_0.15_val_0.15_rescale_8_gamma_0.001/model.joblib"
clf_svm = load(best_model_path_svm)
digits = datasets.load_digits()
image_resized = preprocess(digits.images, 8)
image_resized = np.array(image_resized)
image_resized = image_resized.reshape((len(digits.images), -1))
best_model_path_dc = "./models/test_0.15_val_0.15_rescale_8_maxdepth_14/model.joblib"
clf_dc = load(best_model_path_dc)

###### for Class 0 ######
lst = []
for idx in range(len(digits.images)):
    if digits.target[idx]==0:
        lst.append(idx)
X_test_0 = []
y_test_0 = []
for idx in lst:
    X_test_0.append(image_resized[idx])
    y_test_0.append(digits.target[idx])
predicted_0_svm = clf_svm.predict(X_test_0)
predicted_0_dc = clf_dc.predict(X_test_0)

###### for Class 1 ######
lst = []
for idx in range(len(digits.images)):
    if digits.target[idx]==1:
        lst.append(idx)
X_test_1 = []
y_test_1 = []
for idx in lst:
    X_test_1.append(image_resized[idx])
    y_test_1.append(digits.target[idx])
predicted_1_svm = clf_svm.predict(X_test_1)
predicted_1_dc = clf_dc.predict(X_test_1)

###### for Class 2 ######
lst = []
for idx in range(len(digits.images)):
    if digits.target[idx]==2:
        lst.append(idx)
X_test_2 = []
y_test_2 = []
for idx in lst:
    X_test_2.append(image_resized[idx])
    y_test_2.append(digits.target[idx])
predicted_2_svm = clf_svm.predict(X_test_2)
predicted_2_dc = clf_dc.predict(X_test_2)

###### for Class 3 ######
lst = []
for idx in range(len(digits.images)):
    if digits.target[idx]==3:
        lst.append(idx)
X_test_3 = []
y_test_3 = []
for idx in lst:
    X_test_3.append(image_resized[idx])
    y_test_3.append(digits.target[idx])
predicted_3_svm = clf_svm.predict(X_test_3)
predicted_3_dc = clf_dc.predict(X_test_3)

###### for Class 4 ######
lst = []
for idx in range(len(digits.images)):
    if digits.target[idx]==4:
        lst.append(idx)
X_test_4 = []
y_test_4 = []
for idx in lst:
    X_test_4.append(image_resized[idx])
    y_test_4.append(digits.target[idx])
predicted_4_svm = clf_svm.predict(X_test_4)
predicted_4_dc = clf_dc.predict(X_test_4)

###### for Class 5 ######
lst = []
for idx in range(len(digits.images)):
    if digits.target[idx]==5:
        lst.append(idx)
X_test_5 = []
y_test_5 = []
for idx in lst:
    X_test_5.append(image_resized[idx])
    y_test_5.append(digits.target[idx])
predicted_5_svm = clf_svm.predict(X_test_5)
predicted_5_dc = clf_dc.predict(X_test_5)

###### for Class 6 ######
lst = []
for idx in range(len(digits.images)):
    if digits.target[idx]==6:
        lst.append(idx)
X_test_6 = []
y_test_6 = []
for idx in lst:
    X_test_6.append(image_resized[idx])
    y_test_6.append(digits.target[idx])
predicted_6_svm = clf_svm.predict(X_test_6)
predicted_6_dc = clf_dc.predict(X_test_6)

###### for Class 7 ######
lst = []
for idx in range(len(digits.images)):
    if digits.target[idx]==7:
        lst.append(idx)
X_test_7 = []
y_test_7 = []
for idx in lst:
    X_test_7.append(image_resized[idx])
    y_test_7.append(digits.target[idx])
predicted_7_svm = clf_svm.predict(X_test_7)
predicted_7_dc = clf_dc.predict(X_test_7)

###### for Class 8 ######
lst = []
for idx in range(len(digits.images)):
    if digits.target[idx]==8:
        lst.append(idx)
X_test_8 = []
y_test_8 = []
for idx in lst:
    X_test_8.append(image_resized[idx])
    y_test_8.append(digits.target[idx])
predicted_8_svm = clf_svm.predict(X_test_8)
predicted_8_dc = clf_dc.predict(X_test_8)

###### for Class 9 ######
lst = []
for idx in range(len(digits.images)):
    if digits.target[idx]==9:
        lst.append(idx)
X_test_9 = []
y_test_9 = []
for idx in lst:
    X_test_9.append(image_resized[idx])
    y_test_9.append(digits.target[idx])
predicted_9_svm = clf_svm.predict(X_test_9)
predicted_9_dc = clf_dc.predict(X_test_9)

def test_digit_correct_svm_0():
    assert predicted_0_svm[0]==0

def test_digit_correct_svm_1():
    assert predicted_1_svm[0]==1

def test_digit_correct_svm_2():
    assert predicted_2_svm[0]==2

def test_digit_correct_svm_3():
    assert predicted_3_svm[0]==3

def test_digit_correct_svm_4():
    assert predicted_4_svm[0]==4

def test_digit_correct_svm_5():
    assert predicted_5_svm[1]==5

def test_digit_correct_svm_6():
    assert predicted_6_svm[0]==6

def test_digit_correct_svm_7():
    assert predicted_7_svm[0]==7

def test_digit_correct_svm_8():
    assert predicted_8_svm[0]==8

def test_digit_correct_svm_9():
    assert predicted_9_svm[0]==9

def test_digit_correct_dc_0():    
    assert predicted_0_dc[0]==0

def test_digit_correct_dc_1():
    assert predicted_1_dc[0]==1

def test_digit_correct_dc_2():
    assert predicted_2_dc[0]==2

def test_digit_correct_dc_3():
    assert predicted_3_dc[0]==3

def test_digit_correct_dc_4():
    assert predicted_4_dc[0]==4

def test_digit_correct_dc_5():
    assert predicted_5_dc[1]==5

def test_digit_correct_dc_6():
    assert predicted_6_dc[0]==6

def test_digit_correct_dc_7():
    assert predicted_7_dc[0]==7

def test_digit_correct_dc_8():
    assert predicted_8_dc[0]==8

def test_digit_correct_dc_9():
    assert predicted_9_dc[0]==9