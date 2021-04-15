from tensorflow.python.keras import Sequential
import settings
from tensorflow.keras import models, layers, Model, regularizers
from tensorflow.keras.applications import ResNet50


def CNN3d_LSTM():
    inputs = layers.Input(shape=settings.TENSOR_SHAPE)
    # 3D Convolutional Layers
    x = layers.Conv3D(filters=32, kernel_size=3, padding='same', activation="relu",
                      kernel_regularizer=regularizers.l2(l2=1e-4),
                      bias_regularizer=regularizers.l2(l2=1e-4))(inputs)
    x = layers.MaxPool3D(pool_size=3)(x)
    x = layers.BatchNormalization(center=True, scale=True)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, padding='same', activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      bias_regularizer=regularizers.l2(l2=1e-4))(x)
    x = layers.MaxPool3D(pool_size=3, padding='same')(x)
    x = layers.BatchNormalization(center=True, scale=True)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, padding='same', activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      bias_regularizer=regularizers.l2(l2=1e-4))(x)
    x = layers.MaxPool3D(pool_size=3, padding='same')(x)
    x = layers.BatchNormalization(center=True, scale=True)(x)
    x = layers.Dropout(0.25)(x)

    # LSTM model.
    x = layers.ConvLSTM2D(32, kernel_size=3)(x)
    x = layers.MaxPool2D(pool_size=3, padding='same')(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(units=settings.NUM_CLASSES, activation="softmax")(x)

    # Define the model.
    model = Model(inputs, outputs, name="3dcnn")

    print(model.summary())

    return model
