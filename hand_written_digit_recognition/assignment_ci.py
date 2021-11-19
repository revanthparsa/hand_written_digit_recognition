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
    #print("NO")
else:
    #print("YES")
    pass

lst_train_valid = [(0.15, 0.15)]
class_classifier = [tree.DecisionTreeClassifier, svm.SVC]
for classifier in class_classifier:
    if classifier == tree.DecisionTreeClassifier:
        model_candidates = []
        for test_size, valid_size in lst_train_valid:
            for rescale_factor in rescale_factors:
                print('-'*50)
                for gamma_idx in [exp for exp in range(1, 15)]:
                    image_resized = preprocess(digits.images, rescale_factor)
                    image_resized = np.array(image_resized)
                    image_resized = image_resized.reshape((len(digits.images), -1))

                    X_train, X_valid, X_test, y_train, y_valid, y_test = create_splits(
                        image_resized, digits.target, test_size, valid_size)

                    output_folder = "./models/test_{}_val_{}_rescale_{}_maxdepth_{}".format(
                        test_size, valid_size, rescale_factor, gamma_idx)
                    metric_dic = run_classification_experiment(classifier, X_train, X_valid, 
                                    X_test, y_train, y_valid, y_test, gamma_idx, output_folder)
                    if metric_dic:
                        candidate = {
                            "acc_valid" : metric_dic['acc'],
                            "f1_valid" : metric_dic['f1'],
                            "gamma" : gamma_idx,
                            "rescale_factor": rescale_factor
                        }
                        model_candidates.append(candidate)
                        print('maxdepth: {}, {}x{} ==> {} ==> {}'.format(gamma_idx,
                            rescale_factor, rescale_factor, metric_dic['acc'], metric_dic['f1']))



        max_acc_valid = max(model_candidates, key = lambda x:x["acc_valid"])
        best_model_folder =  "./models/test_{}_val_{}_rescale_{}_maxdepth_{}".format(
            lst_train_valid[0][0], lst_train_valid[0][1], max_acc_valid['rescale_factor'], max_acc_valid['gamma'])
        clf = load(os.path.join(best_model_folder,"model.joblib"))
        metric_dic = test(X_valid, y_valid, clf)
        print('-'*50)
        print('Best depth is {}, Test accuracy {}, F1-Score is {}'.format(max_acc_valid['gamma'], metric_dic['acc'], metric_dic['f1']))
        print('-'*50)

    elif classifier == svm.SVC:
        model_candidates = []
        for test_size, valid_size in lst_train_valid:
            for rescale_factor in rescale_factors:
                print('-'*50)
                for gamma_idx in [10 ** exp for exp in range(-6, 4)]:
                    image_resized = preprocess(digits.images, rescale_factor)
                    image_resized = np.array(image_resized)
                    image_resized = image_resized.reshape((len(digits.images), -1))

                    X_train, X_valid, X_test, y_train, y_valid, y_test = create_splits(
                        image_resized, digits.target, test_size, valid_size)

                    output_folder = "./models/test_{}_val_{}_rescale_{}_gamma_{}".format(
                        test_size, valid_size, rescale_factor, gamma_idx)
                    metric_dic = run_classification_experiment(classifier, X_train, X_valid, 
                                    X_test, y_train, y_valid, y_test, gamma_idx, output_folder)
                    if metric_dic:
                        candidate = {
                            "acc_valid" : metric_dic['acc'],
                            "f1_valid" : metric_dic['f1'],
                            "gamma" : gamma_idx,
                            "rescale_factor": rescale_factor
                        }
                        model_candidates.append(candidate)
                        print('gamma: {}, {}x{} ==> {} ==> {}'.format(gamma_idx,
                            rescale_factor, rescale_factor, metric_dic['acc'], metric_dic['f1']))



        max_acc_valid = max(model_candidates, key = lambda x:x["acc_valid"])
        best_model_folder =  "./models/test_{}_val_{}_rescale_{}_gamma_{}".format(
            lst_train_valid[0][0], lst_train_valid[0][1], max_acc_valid['rescale_factor'], max_acc_valid['gamma'])
        clf = load(os.path.join(best_model_folder,"model.joblib"))
        metric_dic = test(X_valid, y_valid, clf)
        print('-'*50)
        print('Best gamma is {}, Test accuracy {}, F1-Score is {}'.format(max_acc_valid['gamma'], metric_dic['acc'], metric_dic['f1']))
        print('-'*50)
