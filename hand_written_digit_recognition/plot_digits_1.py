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
rescale_factors = [4,8,16]

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
    f1_valid = fl_score(predicted_valid, y_valid , average="macro")
    return {'acc':acc_valid,'f1':f1_valid}
# flatten the images
n_samples = len(digits.images)
data_1 = digits.images
print("Size of the image: {}".format(data_1.shape))
print()
data_1 = data_1.reshape((n_samples, -1))

for test_size, valid_size in [test_size,valid_size]:
    for rescale_factor in rescale_factors:
        for gamma_idx in [10 ** exp for exp in range(-6, 4)]:
            image_resized = preprocess(digits.images, rescale_factor)
            image_resized = image_resized.reshape((n_samples, -1))
            clf1 = svm.SVC(gamma=gamma_idx)
            X_train, X_valid, X_test, y_train, y_valid, y_test = create_splits(
                images, targets, test_size, valid_size)
            

os.mkdir('models')
#For finding the best gamma value


total_len = len(X_train) + len(X_valid) + len(X_test)
print(f'Percentage of samples in Train dataset: {round((len(X_train)/total_len)*100,2)},\
        Valid dataset: {round((len(X_valid)/total_len)*100,2)},\
        Test dataset: {round((len(X_test)/total_len)*100,2)}')
print('-'*50)
gamma_list = []
accuracy_gamma = []
model_candidates = []
for index in range(10):
    gamma_idx = 10**(-6+index)
    gamma_list.append(gamma_idx)
    clf = svm.SVC(gamma = gamma_idx)
    clf.fit(X_train, y_train)

    accuracy_gamma.append(acc_valid)
    if acc_valid < 0.11:
        print("Skipping for gamma {}".format(gamma_idx))
        continue
    candidate = {
        "acc_valid" : acc_valid,
        "gamma" : gamma_idx
    }
    model_candidates.append(candidate)
    output_folder = "./models/test_{}_val_{}_rescale_{}_gamma_{}".format(
        round((len(X_test)/total_len)*100,2), round((len(X_valid)/total_len)*100,2),
        8, gamma_idx)
    os.mkdir(output_folder) 
    dump(clf, os.path.join(output_folder,"model.joblib"))

'''
table = [['S.No','Gamma', 'Validation Accuracy']]
for i in range(len(gamma_list)):
  t = []
  t.append(i+1)
  t.append(gamma_list[i])
  t.append(accuracy_gamma[i])
  table.append(t)
print(tabulate(table, headers='firstrow'))

max_accuracy = max(accuracy_gamma)
max_acc_index = accuracy_gamma.index(max_accuracy)
clf = svm.SVC(gamma = gamma_list[max_acc_index])
clf.fit(X_train, y_train)
'''
max_acc_valid = max(model_candidates,key = lambda x:x["acc_valid"])
best_model_folder =  "./models/test_{}_val_{}_rescale_{}_gamma_{}".format(
        round((len(X_test)/total_len)*100,2), round((len(X_valid)/total_len)*100,2),
        8, max_acc_valid['gamma'])
clf = load(os.path.join(best_model_folder,"model.joblib"))
predicted_test = clf.predict(X_test)
test_accuracy = (accuracy_score(predicted_test, y_test))

print('-'*50)
print('Best gamma is {}, Test accuracy on best gamma is {}'.format(max_acc_valid['gamma'], test_accuracy))
print('-'*50)
