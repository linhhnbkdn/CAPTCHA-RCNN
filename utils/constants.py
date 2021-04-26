import os

DIR = os.path.abspath(os.getcwd())
DATA_FOLDER = os.path.join(DIR, 'data', 'captcha_images_v2dev')

CHAR_LIST = ['d', '3', 'p', 'm', 'b', 'x', 'n', 'y', '7', '4', 'e', 'f',
            'w', 'g', '8', '2', 'c', '5', '6']

BATCH_SIZE = 32