import pandas as pd
import matplotlib.pyplot as plt  
from sklearn import svm
from sklearn import metrics
import joblib
import numpy as np
from sklearn.utils import shuffle

data_set = pd.read_csv("new_data_set.csv")
data_set = data_set.sample(frac=1).reset_index(drop = True)
# print(data_set.head())

x = data_set.drop(["item"],axis = 1)
y = data_set["item"]

x_train = x[0:120]
y_train = y[0:120]

x_test = x[120:]
y_test = y[120:]

model = svm.SVC(kernel = "linear")
model.fit(x_train,y_train)

predict = model.predict(x_test)

print(x_test)
print(predict)

print(metrics.accuracy_score(y_test,predict))