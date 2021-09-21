import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from tabulate import tabulate
import os
from joblib import load, dump

digits = datasets.load_digits()
rescale_factors = [8]
os.mkdir('models')
lst_train_valid = [(0.15, 0.15)]

def preprocess(images, rescale_factor):
    image_resized = []
    for i in range(images.shape[0]):
        image_resized.append(resize(images[i],(rescale_factor,rescale_factor)))
    return image_resized

def create_splits(images, targets, test_size, valid_size):
    X_train, X_test, y_train, y_test = train_test_split(
        images, targets, test_size=test_size + valid_size, shuffle=False)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test, y_test, test_size=valid_size/(test_size + valid_size), shuffle=False)
    return X_train, X_valid, X_test, y_train, y_valid, y_test    

def test(X_valid, y_valid):
    predicted_valid = clf.predict(X_valid)
    acc_valid = accuracy_score(predicted_valid, y_valid)
    f1_valid = f1_score(predicted_valid, y_valid , average="macro")
    return {'acc':acc_valid,'f1':f1_valid}

model_candidates = []
for test_size, valid_size in lst_train_valid:
    for rescale_factor in rescale_factors:
        print('-'*50)
        for gamma_idx in [10 ** exp for exp in range(-6, 4)]:
            image_resized = preprocess(digits.images, rescale_factor)
            image_resized = np.array(image_resized)
            image_resized = image_resized.reshape((len(digits.images), -1))
            clf = svm.SVC(gamma=gamma_idx)
            X_train, X_valid, X_test, y_train, y_valid, y_test = create_splits(
                image_resized, digits.target, test_size, valid_size)
            clf.fit(X_train, y_train)
            metric_dic = test(X_valid, y_valid)
            if metric_dic['acc'] < 0.11:
                print("Skipping for gamma {}".format(gamma_idx))
                continue
            candidate = {
                "acc_valid" : metric_dic['acc'],
                "f1_valid" : metric_dic['f1'],
                "gamma" : gamma_idx,
                "rescale_factor": rescale_factor
            }
            model_candidates.append(candidate)
            output_folder = "./models/test_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, valid_size, rescale_factor, gamma_idx)
            os.mkdir(output_folder) 
            dump(clf, os.path.join(output_folder,"model.joblib"))
            print('gamma: {}, {}x{} ==> {} ==> {}'.format(gamma_idx,
                rescale_factor, rescale_factor, metric_dic['acc'], metric_dic['f1']))


max_acc_valid = max(model_candidates, key = lambda x:x["acc_valid"])
best_model_folder =  "./models/test_{}_val_{}_rescale_{}_gamma_{}".format(
    lst_train_valid[0][0], lst_train_valid[0][1], max_acc_valid['rescale_factor'], max_acc_valid['gamma'])
clf = load(os.path.join(best_model_folder,"model.joblib"))
metric_dic = test(X_valid, y_valid)
print('-'*50)
print('Best gamma is {}, Test accuracy {}, F1-Score is {}'.format(max_acc_valid['gamma'], metric_dic['acc'], metric_dic['f1']))
print('-'*50)
