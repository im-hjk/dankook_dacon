import argparse
import sys

from src.fine_tuning import start_fine_tuning


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
        pass
    elif target_process == 'inference':
        pass