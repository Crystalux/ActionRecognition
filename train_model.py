import models
import settings
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt


def train_3dcnn(train_ds, val_ds):
    # initial_learning_rate = 0.0001
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    # )
    # opt = keras.optimizers.Adam(lr=lr_schedule)
    model = models.cnn3d()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    filepath = './models/3dCNN.h5'
    # callbacks
    cp_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                  save_best_only=True, mode='min',
                                  save_weights_only=True)
    es_callback = EarlyStopping(monitor='accuracy', patience=3)
    lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    callback_list = [es_callback, cp_callback, lr_callback]

    history = model.fit(train_ds, epochs=settings.EPOCHS, batch_size=settings.BATCH_SIZE,
                        callbacks=callback_list, validation_data=val_ds)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.title.set_text('3D CNN loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')

    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.title.set_text('ConvLSTM accuracy')
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')

    fig.savefig('./graphs/3DCNN.png')
    plt.show()


def train_convlstm(train_ds, val_ds):
    opt = keras.optimizers.SGD(lr=0.001)
    model = models.conv_lstm()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    filepath = './models/conv_lstm.h5'
    # callbacks
    cp_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                  save_best_only=True, mode='min',
                                  save_weights_only=True)
    es_callback = EarlyStopping(monitor='accuracy', patience=3)
    lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    callback_list = [es_callback, cp_callback, lr_callback]

    history = model.fit(train_ds, epochs=settings.EPOCHS, batch_size=settings.BATCH_SIZE,
                        callbacks=callback_list, validation_data=val_ds)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.title.set_text('ConvLSTM loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')

    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.title.set_text('ConvLSTM accuracy')
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')

    fig.savefig('./graphs/ConvLSTM.png')
    plt.show()


def train_resnet_lstm(train_ds, val_ds):
    model = models.resnet_lstm()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    filepath = './models/resnet_lstm.h5'
    # callbacks
    cp_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                  save_best_only=True, mode='min',
                                  save_weights_only=True)
    es_callback = EarlyStopping(monitor='accuracy', patience=3)
    lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    callback_list = [es_callback, cp_callback, lr_callback]

    history = model.fit(train_ds, epochs=settings.EPOCHS, batch_size=settings.BATCH_SIZE,
                        callbacks=callback_list, validation_data=val_ds)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.title.set_text('ResNet50-LSTM loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')

    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.title.set_text('ResNet50-LSTM accuracy')
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')

    fig.savefig('./graphs/ResNet50-LSTM.png')
    plt.show()


def train_3dcnn_lstm(train_ds, val_ds):
    # initial_learning_rate = 0.1
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    # )
    # opt = keras.optimizers.SGD(learning_rate=lr_schedule)
    model = models.CNN3d_LSTM()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    filepath = './models/3dCNN_LSTM.h5'
    # callbacks
    cp_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                  save_best_only=True, mode='min',
                                  save_weights_only=True)
    es_callback = EarlyStopping(monitor='accuracy', patience=3)
    lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    callback_list = [es_callback, cp_callback, lr_callback]

    history = model.fit(train_ds, epochs=settings.EPOCHS, batch_size=settings.BATCH_SIZE,
                        callbacks=callback_list, validation_data=val_ds)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.title.set_text('3D CNN-LSTM loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')

    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.title.set_text('3D CNN-LSTM accuracy')
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')

    fig.savefig('./graphs/3dCNN_GRU.png')
    plt.show()

