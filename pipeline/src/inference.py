import os
import sys

from src.data import best_param, divide_data, load_data
from src.train import run_training

sys.path.append(os.path.abspath(os.path.curdir))


def run_inference():
    train_data, test_data = load_data()
    test_X = divide_data(test_data, 'test')

    model = run_training()
    pr = model.predict(test_X)
    return pr


def write_inference(result):
    train_data, test_data = load_data()
    test_data['class'] = result
    test_data.to_csv('./build/inference.csv')


if __name__ == '__main__':
    result = run_inference()
    write_inference(result)