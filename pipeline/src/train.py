import json
import os
import sys

sys.path.append(os.path.abspath(os.path.pardir))
sys.path.append(os.path.abspath(os.path.curdir))

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import optuna

import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.params import SEED
from src.data import best_param, divide_data, load_data


def train_model(model_name, params, X, y):
    if model_name == 'KNeighbor':
        knn_n_neighbors = params.get('knn_n_neighbors', None)
        knn_weights = params.get('knn_n_neighbors', None)
        knn_algorithm = params.get('knn_n_neighbors', None)
        knn_leaf_size = params.get('knn_n_neighbors', None)
        knn_p = params.get('knn_p', None)
        model = KNeighborsClassifier(
            n_neighbors=knn_n_neighbors,
            weights=knn_weights,
            algorithm=knn_algorithm,
            leaf_size=knn_leaf_size,
            p=knn_p
        )

    elif model_name == 'DecisionTree':
        dt_criterion = params.get('dt_criterion', None)
        dt_splitter = params.get('dt_splitter', None)
        model = DecisionTreeClassifier(
            random_state=SEED,
            criterion=dt_criterion,
            splitter=dt_splitter
        )

    elif model_name == 'SVM':
        svm_C = params.get('svm_C', None)
        svm_degree = params.get('svm_degree', None)
        model = SVC(
            random_state=SEED,
            C=svm_C,
            degree=svm_degree
        )

    elif model_name == 'RandomForest':
        rf_max_depth = params.get('rf_max_depth', None)
        rf_max_features = params.get('rf_max_features', None)
        rf_min_samples_leaf = params.get('rf_min_samples_leaf', None)
        rf_n_estimators = params.get('rf_n_estimators', None)
        model = RandomForestClassifier(
            random_state=SEED,
            max_depth=rf_max_depth,
            max_features=rf_max_features,
            min_samples_leaf=rf_min_samples_leaf,
            n_estimators=rf_n_estimators
        )

    elif model_name == 'LightGBM':
        lgbm_max_depth = params.get('lgbm_max_depth', None)
        lgbm_learning_rate = params.get('lgbm_learning_rate', None)
        lgbm_num_leaves = params.get('lgbm_num_leaves', None)
        lgbm_subsample = params.get('lgbm_subsample', None)
        lgbm_lambda_l1 = params.get('lgbm_lambda_l1', None)
        lgbm_lambda_l2 = params.get('lgbm_lambda_l2', None)
        model = lgb.LGBMClassifier(
            random_state=SEED,
            max_depth=lgbm_max_depth,
            num_leaves=lgbm_num_leaves,
            learning_rate=lgbm_learning_rate,
            subsample=lgbm_subsample,
            lambda_l1=lgbm_lambda_l1,
            lambda_l2=lgbm_lambda_l2,
        )

    elif model_name == 'Xgboost':
        xgb_max_depth = params.get('xgb_max_depth', None)
        xgb_learning_rate = params.get('xgb_learning_rate', None)
        model = xgb.XGBClassifier(
            random_state=SEED,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
        )

    else:
        return 'No model selected..'

    print("\nStart training...")
    print(f"Model: {model_name}")

    model_scores = []
    # K-fold CV
    kf = KFold(n_splits=8, random_state=SEED, shuffle=True)
    for train_index, test_index in kf.split(X):
        train_X, test_X = X.iloc[train_index], X.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[train_index]

        model.fit(train_X, train_y)
        print(f"Model ACC: {model.score(test_X, test_y)}%")
        model_scores.append(model.score(test_X, test_y))
    for sc in model_scores:
        print(f'{sc} %')
    print(f"Max ACC: {max(model_scores)}")
    return model


def run_training():
    param = best_param()
    train_data, test_data = load_data()
    train_X, train_y = divide_data(train_data, 'train')
    model = train_model(param['classifier'], param, train_X, train_y)
    return model


if __name__ == '__main__':
    param = best_param()
    train_data, test_data = load_data()
    train_X, train_y = divide_data(train_data, 'train')
    train_model(param['classifier'], param, train_X, train_y)
