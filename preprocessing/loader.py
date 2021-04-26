#!usr/bin env python3
import os
import cv2
import imutils
import logging

import numpy as np

from imutils import paths
from utils import constants


class Loader():
    def __init__(self, dataFolder):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.imgWidth = 200
        self.imgHeight = 50
        self.dataFolder = dataFolder


    def load(self):
        p_imgs = list(paths.list_images(self.dataFolder))
        self.logger.debug("We have: {} images".format(len(p_imgs)))

        imgs = []
        labels = []
        for _img in p_imgs:
            # Get lable for image
            _label = str(_img).split(os.sep)[-1].replace('.png', '')
            _label = self.encode_label(_label)
            # Get grayscale for image
            _img = cv2.imread(_img, cv2.IMREAD_GRAYSCALE)
            _img = imutils.resize(_img, width=self.imgWidth,
                                height=self.imgHeight)
            _img = np.asarray(_img)
            _img = _img/255.
            _img = _img[..., np.newaxis]
            self.logger.debug('Label: {}, IMG: {}'.format(_label, _img.shape))
            imgs.append(_img)
            labels.append(_label)

        self.logger.info("Number of images found: {}".format(len(p_imgs)))
        self.logger.info("Number of labels found: {}".format(len(labels)))

        return (np.asarray(imgs), np.asarray(labels))

    def encode_label(self, label):
        lbl = []
        for char in label:
            lbl.append(constants.CHAR_LIST.index(char))
        return lbl