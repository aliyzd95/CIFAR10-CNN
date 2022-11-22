import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import Sequential
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np


def load_dataset():
    # load dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # one-hot encoding y
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return X_train, y_train, X_test, y_test


# Normalization
def normalize_dataset(train, test):
    train_norm = train.astype('float64')
    test_norm = test.astype('float64')
    # normalize range [0,1]
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm


X_train, y_train, X_test, y_test = load_dataset()
X_train, X_test = normalize_dataset(X_train, X_test)

print(X_train.shape)