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


# flatten the images
n_samples = len(digits.images)
data_1 = digits.images
print("Size of the image: {}".format(data_1.shape))
print()
data_1 = data_1.reshape((n_samples, -1))

'''
data_2 = []
for i in range(data_1.shape[0]):
  data_2.append(resize(data_1[i],(4,4)))
data_2 = np.array(data_2)
print("Size of the image  after resizing: {}".format(data_2.shape))
print()
data_2 = data_2.reshape((n_samples, -1))
data_3 = []
for i in range(data_1.shape[0]):
  data_3.append(resize(data_1[i],(16,16)))
data_3 = np.array(data_3)
print("Size of the image after resizing: {}".format(data_3.shape))
print('-'*50)
data_3 = data_3.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf1 = svm.SVC(gamma=0.001)
clf2 = svm.SVC(gamma=0.001)
clf3 = svm.SVC(gamma=0.001)
clf4 = svm.SVC(gamma=0.001)
clf5 = svm.SVC(gamma=0.001)
clf6 = svm.SVC(gamma=0.001)
clf7 = svm.SVC(gamma=0.001)
clf8 = svm.SVC(gamma=0.001)
clf9 = svm.SVC(gamma=0.001)

# Split data into train and test subsets
X_train_1_1, X_test_1_1, y_train_1_1, y_test_1_1 = train_test_split(
    data_1, digits.target, test_size=0.1, shuffle=False)
X_train_2_1, X_test_2_1, y_train_2_1, y_test_2_1 = train_test_split(
    data_1, digits.target, test_size=0.3, shuffle=False)
X_train_3_1, X_test_3_1, y_train_3_1, y_test_3_1 = train_test_split(
    data_1, digits.target, test_size=0.5, shuffle=False)
X_train_1_2, X_test_1_2, y_train_1_2, y_test_1_2 = train_test_split(
    data_2, digits.target, test_size=0.1, shuffle=False)
X_train_2_2, X_test_2_2, y_train_2_2, y_test_2_2 = train_test_split(
    data_2, digits.target, test_size=0.3, shuffle=False)
X_train_3_2, X_test_3_2, y_train_3_2, y_test_3_2 = train_test_split(
    data_2, digits.target, test_size=0.5, shuffle=False)
X_train_1_3, X_test_1_3, y_train_1_3, y_test_1_3 = train_test_split(
    data_3, digits.target, test_size=0.1, shuffle=False)
X_train_2_3, X_test_2_3, y_train_2_3, y_test_2_3 = train_test_split(
    data_3, digits.target, test_size=0.3, shuffle=False)
X_train_3_3, X_test_3_3, y_train_3_3, y_test_3_3 = train_test_split(
    data_3, digits.target, test_size=0.5, shuffle=False)    


# Learn the digits on the train subset
clf1.fit(X_train_1_1, y_train_1_1)
clf2.fit(X_train_2_1, y_train_2_1)
clf3.fit(X_train_3_1, y_train_3_1)
clf4.fit(X_train_1_2, y_train_1_2)
clf5.fit(X_train_2_2, y_train_2_2)
clf6.fit(X_train_3_2, y_train_3_2)
clf7.fit(X_train_1_3, y_train_1_3)
clf8.fit(X_train_2_3, y_train_2_3)
clf9.fit(X_train_3_3, y_train_3_3)

# Predict the value of the digit on the test subset
predicted_1_1 = clf1.predict(X_test_1_1)
predicted_2_1 = clf2.predict(X_test_2_1)
predicted_3_1 = clf3.predict(X_test_3_1)
predicted_1_2 = clf4.predict(X_test_1_2)
predicted_2_2 = clf5.predict(X_test_2_2)
predicted_3_2 = clf6.predict(X_test_3_2)
predicted_1_3 = clf7.predict(X_test_1_3)
predicted_2_3 = clf8.predict(X_test_2_3)
predicted_3_3 = clf9.predict(X_test_3_3)

print("4x4 --> 0.9-0.1 --> {}".format(accuracy_score(y_test_1_2, predicted_1_2)))
print("4x4 --> 0.7-0.3 --> {}".format(accuracy_score(y_test_2_2, predicted_2_2)))
print("4x4 --> 0.5-0.5 --> {}".format(accuracy_score(y_test_3_2, predicted_3_2)))
print("8x8 --> 0.9-0.1 --> {}".format(accuracy_score(y_test_1_1, predicted_1_1)))
print("8x8 --> 0.7-0.3 --> {}".format(accuracy_score(y_test_2_1, predicted_2_1)))
print("8x8 --> 0.5-0.5 --> {}".format(accuracy_score(y_test_3_1, predicted_3_1)))
print("16x16 --> 0.9-0.1 --> {}".format(accuracy_score(y_test_1_3, predicted_1_3)))
print("16x16 --> 0.7-0.3 --> {}".format(accuracy_score(y_test_2_3, predicted_2_3)))
print("16x16 --> 0.5-0.5 --> {}".format(accuracy_score(y_test_3_3, predicted_3_3)))
print('-'*50)
'''

os.mkdir('models')
#For finding the best gamma value
X_train, X_test, y_train, y_test = train_test_split(
    data_1, digits.target, test_size=0.2, shuffle=False)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_test, y_test, test_size=0.5, shuffle=False)

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
    predicted_valid = clf.predict(X_valid)
    acc_valid =accuracy_score(predicted_valid, y_valid)
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
