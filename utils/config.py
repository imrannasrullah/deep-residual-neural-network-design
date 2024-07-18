# import the necessary packages
import torch
import os
# define path to the original dataset and base path to the dataset
DATA_PATH = "logo_photos"
BASE_PATH = "dataset"
TRAIN_DATA = "data_train.npy"
TRAIN_LABELS = "labels_train.npy"
TEST_DATA = "data_test.npy"
TEST_LABELS = "labels_test.npy"

LABELS_NAMES = ['Nike',
                'Adidas',
                'Ford',
                'Honda',
                'General_mills',
                'Unilever',
                "Mcdonalds",
                'KFC',
                'Gators',
                '3M']

# define validation split and paths to separate train and validation
VAL_SPLIT = 0.3
TRAIN = os.path.join(BASE_PATH, "train")
VAL = os.path.join(BASE_PATH, "val")
TEST = os.path.join(BASE_PATH, "test")

# specify ImageNet mean and standard deviation and image size
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# specify training hyperparameters
FINETUNE_BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001

FINETUNE_PLOT = os.path.join("output", "finetune.png")
FINETUNE_MODEL = os.path.join("output", "finetune_model.pth")