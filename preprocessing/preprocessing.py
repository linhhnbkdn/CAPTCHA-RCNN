#!usr/bin env python3
import cv2
import imutils
import logging

import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from imutils import paths
from utils import constants

from tensorflow.keras import layers



class Preprocessing():
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def get_characters(self):
        return list(set(char for label in self.labels[:] for char in label))

    def preprocess(self):
        XTrain, XTest, YTrain, YTest = train_test_split(self.imgs, self.labels,
                                                    test_size=0.3,
                                                    random_state=42)
        return (XTrain, XTest, YTrain, YTest)
