from tabulate import tabulate
import numpy as np


best_depth = [14, 14, 9, 14, 10]
best_gamma = [0.001, 0.001, 0.001, 0.001, 0.001]
iteration = [0,1,2,3,4]
decision_classifier_accuracy = [0.8407407407407408, 0.8222222222222222, 0.8333333333333334,
								0.8296296296296296, 0.8296296296296296]
decision_classifier_f1_score = [0.8404147745056918, 0.81933529433529432, 0.833404788199957,
								0.8281345155155574, 0.8260112845287862]
svm_accuracy = [0.9925925925925926, 0.9925925925925926, 0.9925925925925926,
				0.9925925925925926, 0.9925925925925926]
svm_f1_score = [0.9926484660849987, 0.9926484660849987, 0.9926484660849987,
				 0.9926484660849987, 0.9926484660849987]

dc_accuracy_mean = str(np.mean(decision_classifier_accuracy))+ ' +/-'
dc_accuracy_std = str(np.std(decision_classifier_accuracy))
dc_f1_score_mean = str(np.mean(decision_classifier_f1_score))+ ' +/-'
dc_f1_score_std = str(np.std(decision_classifier_f1_score))
svm_accuracy_mean = str(np.mean(svm_accuracy))+ ' +/-'
svm_accuracy_std = str(np.std(svm_accuracy))
svm_f1_score_mean = str(np.mean(svm_f1_score))+ ' +/-'
svm_f1_score_std = str(np.std(svm_f1_score))
table = [['Iteration', 'Best Depth', 'Best Gamma', 'DTC Accuracy', 'DTC F1 Score', 'SVM Accuracy', 'SVM F1 Score']]
for i in range(5):
	temp = []
	temp.append(i)
	temp.append(best_depth[i])
	temp.append(best_gamma[i])
	temp.append(decision_classifier_accuracy[i])
	temp.append(decision_classifier_f1_score[i])
	temp.append(svm_accuracy[i])
	temp.append(svm_f1_score[i])
	table.append(temp)

table.append(['','','',dc_accuracy_mean,dc_f1_score_mean,svm_accuracy_mean,svm_f1_score_mean])
table.append(['','','',dc_accuracy_std,dc_f1_score_std,svm_accuracy_std,svm_f1_score_std])

print(tabulate(table))
