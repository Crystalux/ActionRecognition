from tensorflow.python.keras import Sequential

import settings
from tensorflow.keras import models, layers, Model
from tensorflow.keras.applications import ResNet50


def cnn3d():
    inputs = layers.Input(shape=settings.TENSOR_SHAPE)

    x = layers.Conv3D(filters=32, kernel_size=3, padding='same', activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization(center=True, scale=True)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv3D(filters=32, kernel_size=3, padding='same', activation="softmax")(x)
    x = layers.MaxPool3D(pool_size=3, padding='same')(x)
    x = layers.BatchNormalization(center=True, scale=True)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, padding='same', activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2, padding='same')(x)
    x = layers.BatchNormalization(center=True, scale=True)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(units=settings.NUM_CLASSES, activation="softmax")(x)

    # Define the model.
    model = Model(inputs, outputs, name="3dcnn")

    print(model.summary())

    return model


def conv_lstm():
    model = models.Sequential()
    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), return_sequences=False,
                                input_shape=settings.TENSOR_SHAPE))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(settings.NUM_CLASSES, activation='softmax'))

    print(model.summary())

    return model


def resnet_lstm():
    base_model = ResNet50(weights='imagenet', include_top=False)
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    input_layer = layers.Input(shape=settings.TENSOR_SHAPE)

    hidden = layers.TimeDistributed(base_model)(input_layer)
    hidden = layers.TimeDistributed(layers.GlobalAvgPool2D())(hidden)
    hidden = layers.LSTM(1024, activation='relu', return_sequences=False)(hidden)
    hidden = layers.Dense(1024, activation='relu')(hidden)
    hidden = layers.Dropout(0.5)(hidden)
    hidden = layers.Dense(512, activation='relu')(hidden)
    hidden = layers.Dropout(0.5)(hidden)
    hidden = layers.Dense(256, activation='relu')(hidden)
    hidden = layers.Dropout(0.5)(hidden)
    out = layers.Dense(settings.NUM_CLASSES, activation='softmax')(hidden)

    model = Model(inputs=input_layer, outputs=out, name='ResNet-LSTM')

    # model = models.Sequential()
    # model.add(layers.TimeDistributed(base_model))
    # model.add(layers.TimeDistributed(
    #     layers.GlobalAvgPool2D()
    # ))
    # model.add(layers.LSTM(1024, activation='relu', return_sequences=False))
    # model.add(layers.Dense(1024, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(settings.NUM_CLASSES, activation='softmax'))
    #
    print(model.summary())

    return model


def CNN3d_LSTM():
    inputs = layers.Input(shape=settings.TENSOR_SHAPE)
    # 3D Convolutional Layers
    x = layers.Conv3D(filters=32, kernel_size=3, padding='same', activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=3)(x)
    x = layers.BatchNormalization(center=True, scale=True)(x)
    x = layers.Dropout(0.25)(x)
    #
    # x = layers.Conv3D(filters=32, kernel_size=3, padding='same', activation="softmax")(x)
    # x = layers.MaxPool3D(pool_size=3, padding='same')(x)
    # x = layers.BatchNormalization(center=True, scale=True)(x)
    # x = layers.Dropout(0.25)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, padding='same', activation="relu")(x)
    x = layers.MaxPool3D(pool_size=3, padding='same')(x)
    x = layers.BatchNormalization(center=True, scale=True)(x)
    x = layers.Dropout(0.25)(x)


    x = layers.Conv3D(filters=128, kernel_size=3, padding='same', activation="relu")(x)
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
