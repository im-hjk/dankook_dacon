# Fine tuning for DKU Dacon competition

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
from src.data import divide_data, load_data


def objective(trial):
    """
    Fine tuning w. Optuna
    :param trial: optuna study
    :return:
    """

    # load data
    train_X, train_y = divide_data(load_data()[0], 'train')

    classifier = trial.suggest_categorical('classifier', ['KNeighbor', 'DecisionTree', 'SVM',
                                                          'RandomForest', 'LightGBM', 'Xgboost'])

    # KFold
    kf = KFold(n_splits=3, random_state=SEED, shuffle=True)

    if classifier == 'KNeighbor':
        # KNN params
        knn_n_neighbors = trial.suggest_int('n_neighbors', 3, 10)
        knn_weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        knn_algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        knn_leaf_size = trial.suggest_int('leaf_size', 10, 40)
        knn_p = trial.suggest_int('p', 1, 2)

        model = KNeighborsClassifier(
            n_neighbors=knn_n_neighbors,
            weights=knn_weights,
            algorithm=knn_algorithm,
            leaf_size=knn_leaf_size,
            p=knn_p
        )

    elif classifier == 'DecisionTree':
        # DecisionTree params
        dt_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        dt_splitter = trial.suggest_categorical('splitter', ['best', 'random'])

        model = DecisionTreeClassifier(
            random_state=SEED,
            criterion=dt_criterion,
            splitter=dt_splitter
        )

    elif classifier == 'SVM':
        # SVM params
        svm_C = trial.suggest_categorical('svm_C', [0.1, 1, 10, 100, 1000])
        svm_degree = trial.suggest_categorical('svm_degree', [0, 1, 2, 3, 4, 5, 6])

        model = SVC(
            random_state=SEED,
            C=svm_C,
            degree=svm_degree
        )

    elif classifier == 'RandomForest':
        # RandomForest params
        rf_max_depth = trial.suggest_categorical('rf_max_depth', [80, 90, 100, 110])
        rf_max_features = trial.suggest_categorical('rf_max_features', [2, 3])
        rf_min_samples_leaf = trial.suggest_categorical('rf_min_sample_leaf', [8, 10, 12])
        rf_n_estimators = trial.suggest_categorical('rf_n_estimators', [100, 200, 300, 1000])

        model = RandomForestClassifier(
            random_state=SEED,
            max_depth=rf_max_depth,
            max_features=rf_max_features,
            min_samples_leaf=rf_min_samples_leaf,
            n_estimators=rf_n_estimators
        )

    elif classifier == 'LightGBM':
        # LightGBM params
        lgbm_n_estimators = trial.suggest_categorical('lgbm_n_estimators', [100, 500, 1000, 3000])
        lgbm_max_depth = trial.suggest_int('lgbm_max_depth', 20, 200)
        lgbm_learning_rate = trial.suggest_categorical('lgbm_learning_rate', [0.01, 0.05, 0.1])
        lgbm_num_leaves = trial.suggest_categorical('lgbm_num_leaves', [80, 100, 150, 200])
        lgbm_subsample = trial.suggest_categorical('lgbm_subsample', [1, 0.8, 0.7, 0.5])
        lgbm_lambda_l1 = trial.suggest_categorical('lgbm_lambda_l1', [0., 0.5, 0.8, 1])
        lgbm_lambda_l2 = trial.suggest_categorical('lgbm_lambda_l2', [0., 0.5, 0.8, 1])

        model = lgb.LGBMClassifier(
            random_state=SEED,
            max_depth=lgbm_max_depth,
            num_leaves=lgbm_num_leaves,
            learning_rate=lgbm_learning_rate,
            subsample=lgbm_subsample,
            lambda_l1=lgbm_lambda_l1,
            lambda_l2=lgbm_lambda_l2,
        )

    elif classifier == 'Xgboost':
        # Xgboost params
        xgb_max_depth = trial.suggest_int('xgb_max_depth', 10, 200)
        xgb_learning_rate = trial.suggest_categorical('xgb_learning_rate', [0.01, 0.05, 0.1])

        model = xgb.XGBClassifier(
            random_state=SEED,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
        )
        model
    else:
        return

    return cross_val_score(model, train_X, train_y, n_jobs=-1, scoring='accuracy', cv=kf).mean()


def save_best_trial(params):
    with open('./src/best_params.json', 'w') as p_json:
        p_json.write(params)


def start_fine_tuning():
    # run fine tuning
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    trial = study.best_trial
    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

    save_best_trial(trial.params)


if __name__ == '__main__':
    start_fine_tuning()
