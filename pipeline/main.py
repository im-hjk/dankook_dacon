import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.curdir))

from src.fine_tuning import start_fine_tuning
from src.train import run_training
from src.inference import run_inference


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="WA! ReLU's EDA & model trainer")
    parser.add_argument('--process', '-p', type=str, choices=['tuning', 'training', 'inference'],
                        help='target process to execute',
                        )

    args = parser.parse_args()
    target_process = args.process

    if target_process == 'tuning':
        start_fine_tuning()
    elif target_process == 'training':
        run_training()
    elif target_process == 'inference':
        run_inference()