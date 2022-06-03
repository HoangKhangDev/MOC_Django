# LogisticRegression
from importlib.resources import path
import math
import os
import csv
from matplotlib import pyplot as plt
import numpy as np
from numpy import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy import stats
import pylab as pl
import shutil
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import base64


from datetime import datetime
path_stock = os.path.dirname(os.path.abspath("hello"))+os.sep


def Train_batch(batch, path_stock):

    arr = []
    with open(path_stock+"/data.csv") as f:
        arr = f.readlines()
        arr = np.array(arr)
        random.shuffle(arr)
        arr = np.array_split(arr, batch)
    if os.path.exists(path_stock+"/tmp"):
        shutil.rmtree(path_stock+"/tmp")
    os.makedirs(path_stock+"/tmp")
    for i in range(0, len(arr)):
        with open(
                path_stock+f"/tmp/{i+1}.csv", "w") as f:
            f.writelines(arr[i])
    return batch
  # chia thành 10
# n = 20
# bước nhảy k=
# k = 3


def Select_Model(path_open, path_save, n=20, k=3, ChooseModel="GaussianNB"):
    if(ChooseModel != all):
        if ChooseModel == "GaussianNB":
            Model = GaussianNB()
        elif ChooseModel == "DecisionTreeClassifier":
            Model = DecisionTreeClassifier()
        elif ChooseModel == "KNeighborsClassifier":
            Model = KNeighborsClassifier(n_neighbors=8)
        elif ChooseModel == "BernoulliNB":
            Model = BernoulliNB()
        elif ChooseModel == "ExtraTreeClassifier":
            Model = ExtraTreeClassifier()
        elif ChooseModel == "BaggingClassifier":
            Model = BaggingClassifier()
        elif ChooseModel == "AdaBoostClassifier":
            Model = AdaBoostClassifier()
        elif ChooseModel == "MLPClassifier":
            Model = MLPClassifier()
        elif ChooseModel == "LinearDiscriminantAnalysis":
            Model = LinearDiscriminantAnalysis()
        else:
            Model = RandomForestClassifier()
         # read file csv
    data_train = pd.read_csv(path_open).values
    now = datetime.now()

    # convert to string
    str_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    path_save_parent = path_save + \
        f'{ChooseModel}/{str_date}'+os.path.sep
    if(os.path.exists(path_save + f'{ChooseModel}/{str_date}') == False):
        os.makedirs(path_save +
                    f'{ChooseModel}/{str_date}')

    x = data_train[:, 1:11]
    y = data_train[:, 11]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    with open(f"{path_save_parent}/data_train_label.csv", "a") as f:
        f.write(str(y_train))
    with open(f"{path_save_parent}/data_train_value.csv", "a") as f:
        f.write(str(x_train))
    with open(f"{path_save_parent}/data_test_label.csv", "a") as f:
        f.write(str(y_test))
    with open(f"{path_save_parent}/data_test_value.csv", "a") as f:
        f.write(str(x_test))

    begin = 0

    end = int((len(x) * k) / n)
    arr = []
    arr_nu = []
    arr_gamma = []
    x_ve = []
    if(os.path.exists(path_save+f'bieudo') == False):
        os.makedirs(path_save+f'bieudo')
    fs = open(path_save+f'bieudo/{ChooseModel}.csv', "w")
    fs.write("")
    fs = open(path_save+f'bieudo/{ChooseModel}.csv', "a")
    for i in range(n):
        trainx = x[begin:end]
        trainy = y[begin:end]
        # model = func.fit(trainx, trainy)
        Model.fit(trainx, trainy)

        y_pred = Model.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        begin += int(len(x_train) / n)
        end += int(len(x_train) / n)
        arr.append(accuracy)
        x_ve.append(i+1)
        arr_nu.append(trainx)
        arr_gamma.append(trainy)
        fs.write(f"{i} {accuracy}\n")
    shutil.rmtree(path_save_parent)
    shutil.rmtree(path_save+'/bieudo')
    return arr_nu, arr_gamma, arr, x_ve
