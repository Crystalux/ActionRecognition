from tensorflow.python.keras import Sequential
import settings
from tensorflow.keras import models, layers, Model, regularizers
from tensorflow.keras.applications import ResNet50


def cnn3d(stand_alone=True):
    inputs = layers.Input(shape=settings.TENSOR_SHAPE, name='rgb')
    # 3D Convolutional Layers
    x = layers.Conv3D(filters=64, kernel_size=3, strides=2, padding='same', activation="relu",
                      kernel_regularizer=regularizers.l2(l2=1e-4),
                      bias_regularizer=regularizers.l2(l2=1e-4))(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, padding='same', strides=1, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      bias_regularizer=regularizers.l2(l2=1e-4))(x)
    x = layers.MaxPool3D(pool_size=2, padding='same')(x)


    x = layers.Conv3D(filters=256, kernel_size=3, strides=1, padding='same', activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      bias_regularizer=regularizers.l2(l2=1e-4))(x)
    x = layers.MaxPool3D(pool_size=2, padding='same')(x)

    x = layers.Conv3D(filters=256, kernel_size=3, strides=1, padding='same', activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      bias_regularizer=regularizers.l2(l2=1e-4))(x)
    x = layers.MaxPool3D(pool_size=2, padding='same')(x)


    x = layers.Flatten()(x)
    if stand_alone:
        x = layers.Dense(units=512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

        outputs = layers.Dense(units=settings.NUM_CLASSES, activation="softmax")(x)
    else:
        x = layers.Dense(units=2048, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(units=1024, activation="relu")(x)

    # Define the model.
    model = Model(inputs, outputs, name="3dcnn")

    return model


def cnn2d_lstm(stand_alone=True):
    inputs = layers.Input(shape=settings.TENSOR_SHAPE, name='optical')
    x = layers.TimeDistributed(layers.Conv2D(32, kernel_size=3, strides=2,
                                             padding='same', activation="relu",
                                             kernel_regularizer=regularizers.l2(1e-4),
                                             bias_regularizer=regularizers.l2(l2=1e-4)))(inputs)
    x = layers.TimeDistributed(layers.MaxPool2D(pool_size=2, padding='same'))(x)

    x = layers.TimeDistributed(layers.Conv2D(64, kernel_size=5, strides=2, padding='same', activation="relu",
                                             kernel_regularizer=regularizers.l2(1e-4),
                                             bias_regularizer=regularizers.l2(l2=1e-4)))(x)
    x = layers.TimeDistributed(layers.MaxPool2D(pool_size=2, padding='same'))(x)

    x = layers.TimeDistributed(layers.Flatten())(x)

    # x = layers.LSTM(256)(x)
    x = layers.LSTM(1024)(x)

    if stand_alone:
        x = layers.Dense(units=512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(units=settings.NUM_CLASSES, activation="softmax")(x)
    else:
        outputs = layers.Dense(units=1024, activation="relu")(x)

    model = Model(inputs, outputs, name = '2D CNN-LSTM')

    return model


def ts_3dcnn():
    rgb_cnn = cnn3d(stand_alone=False)
    opt_cnn = cnn2d_lstm(stand_alone=False)

    ts = layers.concatenate([rgb_cnn.output, opt_cnn.output])

    x = layers.Dropout(0.5)(ts)
    x = layers.Dense(units=1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=settings.NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=[rgb_cnn.input, opt_cnn.inputs], outputs=x, name="TS-3DCNN")

    return model


