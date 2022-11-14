import pandas as pd
from xgboost import plot_importance as xgb_plot
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from lightgbm import LGBMClassifier
from lightgbm import plot_importance as lgb_plot
import warnings
import matplotlib.pyplot as plt
import tempfile
import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import argparse
from callbacks_custom import Plotting, CustomLearningRate

warnings.filterwarnings(action='ignore')

# plt.rc('font', family='NanumBarunGothic') 
# warnings.simplefilter("ignore")

'''Demo for defining a custom callback function that plots evaluation result during
training.'''
X, y = load_breast_cancer(return_X_y=True)
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0, stratify=y)

# D_train = xgb.DMatrix(X_train, y_train)
# D_valid = xgb.DMatrix(X_valid, y_valid)

num_boost_round = 10000
# plotting = Plotting(num_boost_round)
lr_scheduler = CustomLearningRate()
# Pass it to the `callbacks` parameter as a list.
model = xgb.XGBClassifier(n_estimators=num_boost_round)

model.fit(X_train, y_train,eval_set=[(X_valid, y_valid)], early_stopping_rounds=400, eval_metric=['rmse', 'auc'], callbacks=[lr_scheduler], verbose=False)
# model.fit(X_train, y_train,eval_set=[(X_valid, y_valid)],  eval_metric=['rmse', 'auc'], )