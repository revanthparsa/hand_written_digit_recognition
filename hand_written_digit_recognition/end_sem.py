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
classifier = tree.DecisionTreeClassifier
temp = []    
lst_1 = []
print('Maxdepth\t Train Accuracy\t Train F1\t Valid Accurtacy\tValid F1\tTest Accurtacy\tTest F1')
for NumTimes in range(3):
    model_candidates = []
    for test_size, valid_size in lst_train_valid:
        for rescale_factor in rescale_factors:
            #print('-'*50)
            lst_temp = []
            for gamma_idx in [exp for exp in range(1, 15)]:
                lst = []
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
                        "acc_valid" : metric_dic['acc_valid'],
                        "f1_valid" : metric_dic['f1_valid'],
                        "acc_train" : metric_dic['acc_train'],
                        "f1_train" : metric_dic['f1_train'],
                        "gamma" : gamma_idx,
                        "rescale_factor": rescale_factor
                    }
                    model_candidates.append(candidate)
                    #print('maxdepth: {}, {}x{} ==> {} ==> {} ==> {} ==> {}'.format(gamma_idx,
                    # #   rescale_factor, rescale_factor, metric_dic['acc_valid'], metric_dic['f1_valid'],
                    #    metric_dic['acc_train'], metric_dic['f1_train']))
                    lst.append(metric_dic['acc_valid'])
                    lst.append(metric_dic['f1_valid'])
                    lst.append(metric_dic['acc_train'])
                    lst.append(metric_dic['f1_train'])
                lst_temp.append(lst)
    #temp.append(lst_temp)                
    max_acc_valid = max(model_candidates, key = lambda x:x["acc_valid"])
    best_model_folder =  "./models/test_{}_val_{}_rescale_{}_maxdepth_{}".format(
        lst_train_valid[0][0], lst_train_valid[0][1], max_acc_valid['rescale_factor'], max_acc_valid['gamma'])
    clf = load(os.path.join(best_model_folder,"model.joblib"))
    metric_dic_1 = test(X_valid, y_valid, X_train,y_train, clf)
    metric_dic_2 = test(X_test, y_test, X_train,y_train, clf)
    #print('-'*50)
    print('{}\t {}\t {}\t {}\t{}\t {}\t{}'.format(max_acc_valid['gamma'], metric_dic_1['acc_train'], metric_dic_1['f1_train'], metric_dic_1['acc_valid'], metric_dic_1['f1_valid'], metric_dic_2['acc_valid'], metric_dic_2['f1_valid']))