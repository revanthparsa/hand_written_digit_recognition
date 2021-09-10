import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from tabulate import tabulate

digits = datasets.load_digits()


# flatten the images
n_samples = len(digits.images)
print("Size of the image: {}".format(digits.images.shape))
data_1 = digits.images.reshape((n_samples, -1))


# Create a classifier: a support vector classifier
clf1 = svm.SVC(gamma=0.001)
clf2 = svm.SVC(gamma=0.001)
clf3 = svm.SVC(gamma=0.001)
# Split data into 50% train and 50% test subsets
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    data_1, digits.target, test_size=0.1, shuffle=False)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    data_1, digits.target, test_size=0.3, shuffle=False)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
    data_1, digits.target, test_size=0.5, shuffle=False)
'''
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    data_2, digits.target, test_size=0.1, shuffle=False)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    data_2, digits.target, test_size=0.3, shuffle=False)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
    data_2, digits.target, test_size=0.5, shuffle=False)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    data_3, digits.target, test_size=0.1, shuffle=False)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    data_3, digits.target, test_size=0.3, shuffle=False)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
    data_3, digits.target, test_size=0.5, shuffle=False)    
'''
# Learn the digits on the train subset
clf1.fit(X_train_1, y_train_1)
clf2.fit(X_train_2, y_train_2)
clf3.fit(X_train_3, y_train_3)
# Predict the value of the digit on the test subset
predicted_1 = clf1.predict(X_test_1)
predicted_2 = clf2.predict(X_test_2)
predicted_3 = clf3.predict(X_test_3)

print("8x8 --> 0.9-0.1 --> {}".format(accuracy_score(y_test_1, predicted_1)))
print("8x8 --> 0.7-0.3 --> {}".format(accuracy_score(y_test_2, predicted_2)))
print("8x8 --> 0.5-0.5 --> {}".format(accuracy_score(y_test_3, predicted_3)))


gamma_list = []
accuracy_gamma = []
for idx in range(1,20):
  gamma_list.append(0.001*idx)
  clf1 = svm.SVC(gamma=0.001*idx)
  clf1.fit(X_train_1, y_train_1)
  predicted_1 = clf1.predict(X_test_1)
  accuracy_gamma.append(accuracy_score(y_test_1, predicted_1))

print('.'*50)
print('Accuracy of model varying the gamma')
print('Gamma varies from 0.001 to 0.02 in steps of 0.001')
print(accuracy_gamma)
print('.'*50)

plt.plot(gamma_list,accuracy_gamma)
plt.xlabel('gamma')
plt.ylabel('accuracy')
print('.'*50)
print('Accuracy decreases as the gamma increases')
print('.'*50)

table = [['S.No','Gamma', 'Accuracy']]
for i in range(len(gamma_list)):
  t = []
  t.append(i+1)
  t.append(gamma_list[i])
  t.append(accuracy_gamma[i])
  table.append(t)
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
