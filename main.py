import os

import torch

from train.yolo_trainer import YOLOTrainer

absolute_path = os.path.abspath(__file__)


def main():

    trainer = YOLOTrainer()

    trainer.train()

    trainer.evaluate()

    trainer.save_model()


if __name__ == "__main__":
    main()
