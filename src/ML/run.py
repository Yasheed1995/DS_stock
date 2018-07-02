from __future__ import print_function
import sys
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.preprocessing import StandardScaler
from util import DataManager
from plot import plot_conf_matrix, plot_curve
import numpy as np
import pandas as pd
import pickle
import os

def model_generator(token):
    if token == 'LinR':
        model = LinearRegression(normalize=True) 
    elif token == 'LogR':
        model = LogisticRegression()
    elif token == 'SVM':
        model = SVC() 
    elif token == 'D-Tree':
        model = DecisionTreeClassifier() 
    elif token == 'NN':
        model = MLPClassifier() 
    elif token == 'RF':
        model = RandomForestClassifier(n_estimators=1)
    elif token == 'KMeans':
        model = KMeans()
    elif token == 'Bayes':
        model = GaussianNB()
    return model

def train(data, label, token, bin_size=None): 
    if token == 'LinR':
        train_data = data[:-60]
        train_label = label[:-60]
        test_data = data[-60:]
        test_label = label[-60:]
        train_label = np.expand_dims(train_label, axis=1)
        test_label = np.expand_dims(test_label, axis=1)
        model = model_generator(token)
        model.fit(train_data, train_label)
        modeldir = os.path.join('models', 'LinR') 
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
        pickle.dump(model, open(os.path.join(modeldir, 'model.pkl'), 'wb'))
        predict = model.predict(test_data)
        figdir = os.path.join('fig', token)
        if not os.path.exists(figdir):
            os.makedirs(figdir)
        plot_curve(predict, test_label, token, os.path.join(figdir, 'curve.pdf'))
        MSE = mean_squared_error(test_label, predict)
        MAE = mean_absolute_error(test_label, predict)
        return MSE, MAE
    else:
        train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2)
        min_val = min(train_label)
        max_val = max(train_label)
        bins = [ min_val + idx * (max_val - min_val) / (bin_size - 1) for idx in range(bin_size)]
        labels = range(bin_size-1)
        train_label = pd.cut(train_label, bins=bins, labels=labels)
        test_label = pd.cut(test_label, bins=bins, labels=labels)
        model = model_generator(token)
        for i in range(len(train_label)):
            if train_label[i] != train_label[i]:
                train_label[i] = 0
        for i in range(len(test_label)):
            if test_label[i] != test_label[i]:
                test_label[i] = 0
        model.fit(train_data, train_label)
        modeldir = os.path.join('models', token) 
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
        pickle.dump(model, open(os.path.join(modeldir, 'bins-{}.pkl'.format(bin_size)), 'wb'))
        predict = model.predict(test_data)
        conf_matrix = confusion_matrix(test_label, predict, labels=labels)
        figdir = os.path.join('fig', token)
        if not os.path.exists(figdir):
            os.makedirs(figdir)
        figpath = os.path.join(figdir, 'bins-{}.pdf'.format(bin_size))
        plot_conf_matrix(conf_matrix, labels, True, token, figpath)
        accuracy = accuracy_score(test_label, predict)
        precision, recall, f, _ = precision_recall_fscore_support(test_label, predict, average='weighted')
        return accuracy, precision, recall, f

def argument_parser(L):
    token = L[1]
    dm = DataManager()
    dm.add_data('data/data.csv')
    X = dm.get_data('data')
    Y = dm.get_data('label')
    data = X[0]
    label = Y[0]
    logpath = os.path.join('log')
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    if token == 'LinR':
        MSE, MAE = train(data, label, token)
        with open('log/LinR.csv', 'w') as f:
            f.write('MSE,MAE\n')
            f.write('{},{}\n'.format(MSE, MAE))
    else:
        bin_size = int(L[2])
        acc, pre, rec, f_score = train(data, label, token, bin_size=bin_size)
        with open('log/' + token + '-bins-' + str(bin_size) + '.csv', 'w') as f:
            f.write('accuracy,precision,recall,f-score\n')
            f.write('{},{},{},{}\n'.format(acc, pre, rec, f_score))

if __name__ == '__main__':
    argument_parser(sys.argv)
