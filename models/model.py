from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from models.CTCLayer import CTCLayer

class RCNN_CTCLoss():
    @classmethod
    def build(cls, height, width, depth, characters):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential(name='RCNN_CTCLoss')
        inputShape = (height, width, depth)
        chanDim = -1
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        # Input block
        inputImg = Input(shape=inputShape, name="image", dtype="float32")
        labels = Input(name="label", shape=(None, ), dtype="float32")
        # First conv block
        seq = Conv2D(32, (3, 3), activation='relu',
                        kernel_initializer='he_normal',
                        padding='same',
                        name='Conv1',
                        input_shape=inputShape)(inputImg)
        seq = MaxPooling2D((2, 2), name="pool1")(seq)
        # Second conv block
        seq = Conv2D(64, (3, 3), activation='relu',
                        kernel_initializer="he_normal",
                        padding="same",
                        name='Conv2')(seq)
        seq = MaxPooling2D((2, 2), name='pool2')(seq)
        seq = Reshape(target_shape=(50, -1))(seq)
        # RNNs
        seq = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(seq)
        seq = Bidirectional(LSTM(64, return_sequences=True, dropout=0.25))(seq)
        # Output layers
        seq = Dense(len(characters) + 1, activation='softmax', name='dense1')(seq)
        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name='ctc_loss')(labels, seq)
        # Define the model
        model = Model(inputs=[inputImg, labels], outputs=output, name='RCNN_CTCLoss')
        # # Optimizer
        opt = Adam(learning_rate=0.001)
        # # Compile the model and return
        model.compile(optimizer=opt)
        model.summary()
        return model