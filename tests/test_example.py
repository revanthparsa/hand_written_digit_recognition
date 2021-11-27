import math
from sklearn import datasets, svm, metrics, tree
import os
from joblib import load, dump
import numpy as np
import sys  
sys.path.append('/home/revanth/hand_written_digit_recognition/hand_written_digit_recognition') 
#from utils import preprocess, create_splits, run_classification_experiment
from sklearn.model_selection import train_test_split
def create_splits(images, targets, test_size, valid_size):
    X_train, X_test, y_train, y_test = train_test_split(
        images, targets, test_size=test_size + valid_size, shuffle=False)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test, y_test, test_size=valid_size/(test_size + valid_size), shuffle=False)
    return X_train, X_valid, X_test, y_train, y_valid, y_test
'''
def test_model_writing():

    ##1. create some data
    digits = datasets.load_digits()
    rescale_factor = 16
    test_size = 0.15
    valid_size = 0.15
    gamma_idx = 0.001
    image_resized = preprocess(digits.images, rescale_factor)
    image_resized = np.array(image_resized)
    image_resized = image_resized.reshape((len(digits.images), -1))
    X_train, X_valid, X_test, y_train, y_valid, y_test = create_splits(
        image_resized, digits.target, test_size, valid_size)

    output_folder = "/home/r/hand_written_digit_recognition/hand_written_digit_recognition/models/test_{}_val_{}_rescale_{}_gamma_{}".format(
        test_size, valid_size, rescale_factor, gamma_idx)

    ##2. run_classification_experiment(data, expeted-model-file)
    run_classification_experiment(svm.SVC, X_train, X_valid, 
                        X_test, y_train, y_valid, y_test, gamma_idx, output_folder)
    final_folder = output_folder + '/model.joblib'
    assert os.path.isfile(final_folder)


def test_small_data_overfit_checking():

    #1. create a small amount of data / (digits / subsampling)
    digits = datasets.load_digits()
    rescale_factor = 16
    test_size = 0.15
    valid_size = 0.15
    gamma_idx = 0.001
    image_resized = preprocess(digits.images, rescale_factor)
    image_resized = np.array(image_resized)
    image_resized = image_resized.reshape((len(digits.images), -1))
    X_train, X_valid, X_test, y_train, y_valid, y_test = create_splits(
        image_resized, digits.target, test_size, valid_size)

    output_folder = "/home/r/hand_written_digit_recognition/hand_written_digit_recognition/models/test_{}_val_{}_rescale_{}_gamma_{}".format(
        test_size, valid_size, rescale_factor, gamma_idx)

    #2. train_metrics = run_classification_experiment(train=train, valid=train)
    train_metrics =  run_classification_experiment(svm.SVC, X_valid, X_valid, 
                            X_test, y_valid, y_valid, y_test, gamma_idx, output_folder)

    #assert train_metrics['acc']  > some threshold
    assert train_metrics['acc']  > 0.99
    #assert train_metrics['f1'] > some other threshold
    assert train_metrics['f1'] > 0.99
'''

def test_create_splits():
    digits = datasets.load_digits()
    n_samples = 100
    num_train = 70
    num_test = 20
    num_valid = 10
    test_size = 0.10
    valid_size = 0.199
    digits.images = digits.images[:n_samples]
    digits.target = digits.target[:n_samples]
    image_resized = np.array(digits.images)
    image_resized = image_resized.reshape((len(digits.images), -1))
    X_train, X_valid, X_test, y_train, y_valid, y_test = create_splits(
        image_resized, digits.target, test_size, valid_size)
    assert len(X_train) == num_train
    assert len(X_test) == num_test
    assert len(X_valid) == num_valid
    assert len(X_train)+len(X_valid)+ len(X_test) == n_samples   

    n_samples = 9
    num_train = 6
    num_test = 2
    num_valid = 1
    test_size = 0.1
    valid_size = 0.2
    digits.images = digits.images[:n_samples]
    digits.target = digits.target[:n_samples]
    image_resized = np.array(digits.images)
    image_resized = image_resized.reshape((len(digits.images), -1))
    X_train, X_valid, X_test, y_train, y_valid, y_test = create_splits(
        image_resized, digits.target, test_size, valid_size)
    assert len(X_train) == num_train
    assert len(X_test) == num_test
    assert len(X_valid) == num_valid
    assert len(X_train)+len(X_valid)+ len(X_test) == n_samples  