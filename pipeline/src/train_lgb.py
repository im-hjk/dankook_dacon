# LightGBM

import logging
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from pipeline.src.params import N_FOLD, N_CLASS, SEED


def train_predict(train_file, test_file, feature_map_file,
                  predict_valid_file, predict_test_file, feature_imp_file,
                  n_est=100, n_leaf=200, lrate=.1, n_min=8, sub_col=.3,
                  sub_row=.8, sub_row_freq=100, n_stop=100):
    MODEL_NAME = 'lightGBM'

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.DEBUG, filemode=f'{MODEL_NAME}.log')

    logging.info('Loading train/test data..')
