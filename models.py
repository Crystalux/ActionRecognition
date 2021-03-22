import settings
from tensorflow.keras import models, layers


def conv_lstm():
    model = models.Sequential()
    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(3,3), return_sequences=False,
                                input_shape=settings.TENSOR_SHAPE))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(settings.NUM_CLASSES, activation='softmax'))

    print(model.summary())

    return model
