import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm, metrics, tree
from sklearn.metrics import accuracy_score, f1_score
import os
from joblib import load, dump
from utils import preprocess, create_splits, test, run_classification_experiment

digits = datasets.load_digits()
rescale_factors = [8]

if not (os.path.exists("./models")):
    os.mkdir('models')
else:
    pass

lst_train_valid = [(0.1, 0.1)]
print('#'*20+ ' SVMClassifier '+ '#'*20)
classifier = svm.SVC
model_candidates = [[] for i in range(10)]
for test_size, valid_size in lst_train_valid:
    for rescale_factor in rescale_factors:
        lst_acc_svm = [[] for i in range(10)]
        lst_f1_svm = [[] for i in range(10)]
        for gamma_idx in [10 ** exp for exp in range(-6, 4)]:
            #print('-'*50)
            image_resized = preprocess(digits.images, rescale_factor)
            image_resized = np.array(image_resized)
            image_resized = image_resized.reshape((len(digits.images), -1))

            X_train, X_valid, X_test, y_train, y_valid, y_test = create_splits(
                image_resized, digits.target, test_size, valid_size)
            
            for idx in range(1,11):
                output_folder = "./models/train_{}_test_{}_val_{}_rescale_{}_gamma_{}".format(
                    idx*10, test_size, valid_size, rescale_factor, gamma_idx)
                metric_dic = run_classification_experiment(classifier, X_train[:int(0.1*idx*len(X_train))+1], X_valid, 
                                X_test, y_train[:int(0.1*idx*len(y_train))+1], y_valid, y_test, gamma_idx, output_folder)
                candidate = {
                    "acc_valid" : metric_dic['acc'],
                    "f1_valid" : metric_dic['f1'],
                    "gamma" : gamma_idx,
                    "rescale_factor": rescale_factor
                }
                model_candidates[idx-1].append(candidate)
                #print('train_size: {}, gamma: {}, {}x{} ==> {} ==> {}'.format(idx*10, gamma_idx,
                #    rescale_factor, rescale_factor, metric_dic['acc'], metric_dic['f1']))
                lst_acc_svm[idx-1].append(round(metric_dic['acc'],4))
                lst_f1_svm[idx-1].append(round(metric_dic['f1'],4))

lst_best_f1_svm = []
for idx in range(1,11):
    max_acc_valid = max(model_candidates[(idx-1)], key = lambda x:x["acc_valid"])
    best_model_folder =  "./models/train_{}_test_{}_val_{}_rescale_{}_gamma_{}".format(
        idx*10, lst_train_valid[0][0], lst_train_valid[0][1], max_acc_valid['rescale_factor'], max_acc_valid['gamma'])
    clf = load(os.path.join(best_model_folder,"model.joblib"))
    metric_dic = test(X_valid, y_valid, clf)
    print('-'*50)
    lst_best_f1_svm.append(metric_dic['f1'])
    print('Traindata Size: {}%, Best gamma is {}, Test accuracy {}, F1-Score is {}'.format((idx)*10, max_acc_valid['gamma'], metric_dic['acc'], metric_dic['f1']))
print('-'*50)
print('#'*20+ ' DecisionTreeClassifier '+ '#'*20)
classifier = tree.DecisionTreeClassifier
model_candidates = [[] for i in range(10)]
for test_size, valid_size in lst_train_valid:
    for rescale_factor in rescale_factors:
        lst_acc_dc = [[] for i in range(10)]
        lst_f1_dc = [[] for i in range(10)]
        for gamma_idx in [exp for exp in range(1, 15)]:
            #print('-'*50)
            image_resized = preprocess(digits.images, rescale_factor)
            image_resized = np.array(image_resized)
            image_resized = image_resized.reshape((len(digits.images), -1))

            X_train, X_valid, X_test, y_train, y_valid, y_test = create_splits(
                image_resized, digits.target, test_size, valid_size)
            
            for idx in range(1,11):
                output_folder = "./models/train_{}_test_{}_val_{}_rescale_{}_maxdepth_{}".format(
                    idx*10, test_size, valid_size, rescale_factor, gamma_idx)
                metric_dic = run_classification_experiment(classifier, X_train[:int(0.1*idx*len(X_train))+1], X_valid, 
                                X_test, y_train[:int(0.1*idx*len(y_train))+1], y_valid, y_test, gamma_idx, output_folder)
                candidate = {
                    "acc_valid" : metric_dic['acc'],
                    "f1_valid" : metric_dic['f1'],
                    "gamma" : gamma_idx,
                    "rescale_factor": rescale_factor
                }
                model_candidates[idx-1].append(candidate)
                #print('train_size: {}, maxdepth: {}, {}x{} ==> {} ==> {}'.format(idx*10, gamma_idx,
                #    rescale_factor, rescale_factor, metric_dic['acc'], metric_dic['f1']))
                lst_acc_dc[idx-1].append(round(metric_dic['acc'],4))
                lst_f1_dc[idx-1].append(round(metric_dic['f1'],4))

lst_best_f1_dc = []
for idx in range(1,11):
    max_acc_valid = max(model_candidates[(idx-1)], key = lambda x:x["acc_valid"])
    best_model_folder =  "./models/train_{}_test_{}_val_{}_rescale_{}_maxdepth_{}".format(
        idx*10, lst_train_valid[0][0], lst_train_valid[0][1], max_acc_valid['rescale_factor'], max_acc_valid['gamma'])
    clf = load(os.path.join(best_model_folder,"model.joblib"))
    metric_dic = test(X_valid, y_valid, clf)
    print('-'*50)
    lst_best_f1_dc.append(metric_dic['f1'])
    print('Traindata Size: {}%, Best maxdepth is {}, Test accuracy {}, F1-Score is {}'.format((idx )*10, max_acc_valid['gamma'], metric_dic['acc'], metric_dic['f1']))
print('-'*50)
plt.plot([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], lst_best_f1_svm, label = 'SVMClassifier')
plt.plot([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], lst_best_f1_dc, label = 'DecisionTreeClassifier')
plt.title('Training dataset percentage vs Macro F1 Score')
plt.xlabel('Training dataset Percentage')
plt.ylabel('Test Macro F1 Score')
plt.legend()
plt.show()