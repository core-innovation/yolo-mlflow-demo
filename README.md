# YOLO Model Training Repository

## Overview
This repository provides a clean, reproducible, and well-documented codebase for training any version of YOLO (e.g., YOLOv5, YOLOv8n, YOLOv8m) by simply changing the dataset and configurations. You can modify the device used for training (e.g., CPU or GPU) through the 'device' setting in `config.yaml`.

## Features
- Object-oriented design for better code structure.
- Easy configuration through `config/config.yaml`.
- Supports multiple YOLO versions.
- Tracks experiments through MLFlow.
- Data preprocessing tools for YOLO format.
- Custom logging and evaluation.


## Code
`train/yolo_trainer.py`: Contains the `YOLOTrainer` class that encapsulates the YOLO training logic. The class initializes the model based on the config.yaml file, trains the model, and starts an MLFlow experiment. Training parameters such as epochs, img_size, batch_size, and device are retrieved from the config.yaml file. The model is also evaluated and saved after training.


## Installation
```bash
pip install -r requirements.txt
```

## Adding Data
Create a folder called data inside the data directory, and within it, add **train**, **val**, and **test** folders containing images in the YOLO format. The folder structure should look like this:
```
─data
│──data
│  │──train
│  │  |──images
│  │  |──labels
│  │──val
│  │  |──images
│  │  |──labels
│  │──test
│  │  |──images
│  │  |──labels
│──dataset.yaml
```

## Modifying Configurations
Change the **config.yaml** and **data.yaml** by adding the paths to the data, the model that you want to use, the classes and other features that may apply to your project. 

## Running the Code
- Open mlflow server
```bash
mlflow ui 
```
- Run code
```bash
python main.py
```
- See experiment and compare at the suggested url.

