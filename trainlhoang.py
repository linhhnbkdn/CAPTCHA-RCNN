import os

from utils import constants
from preprocessing import Loader
from preprocessing import Preprocessing

from models.model import RCNN_CTCLoss

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

loader  = Loader(constants.DATA_FOLDER)
imgs, labels = loader.load()

preproc = Preprocessing(imgs, labels)
XTrain, XTest, YTrain, YTest = preproc.preprocess()


model = RCNN_CTCLoss.build(50, 200, 1, preproc.get_characters())

epochs = 10
modelPath = os.path.join('./', '{}.h5'.format(model.__class__.__name__))
checkpoint = ModelCheckpoint(modelPath, monitor='val_accuracy',
                            verbose=1, save_best_only=True,
                            mode='min')

# Train the model
history = model.fit(
    (XTrain, YTrain),
    validation_data=[XTest, YTest],
    epochs=epochs,
    callbacks=[checkpoint],
    batch_size=constants.BATCH_SIZE,
)
