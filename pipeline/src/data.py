import json
import os
import sys

sys.path.append(os.path.abspath(os.path.pardir))
sys.path.append(os.path.abspath(os.path.curdir))

import pandas as pd

from src.params import DATA_PATH


def load_data():
    # load train, test data
    train_data = pd.read_csv(f'{DATA_PATH}/train.csv')
    test_data = pd.read_csv(f'{DATA_PATH}/test.csv')

    print(f"train data: {len(train_data)}")
    print(f"test data: {len(test_data)}")

    return train_data, test_data


def divide_data(data, dtype='train'):
    # Separate X, y from data, return whole data if test
    if dtype == 'train':
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1:]['class']

        print(f"train X \n {X.describe()}")
        print(f"train y \n {y.describe()}")

        return X, y
    elif dtype == 'test':
        print(f"test X \n {data.describe()}")
        return data
    else:
        return None


def best_param(path='./src/best_params.json'):
    if os.path.exists(path):
        with open(path, 'r') as p_json:
            param = json.load(p_json)
        return param


if __name__ == '__main__':
    param = best_param('../src/best_params.json')
    print(param)