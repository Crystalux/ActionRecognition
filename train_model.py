import models
import settings
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt


def train_3dcnn(train, val,  dataset):
    train_rgb = train.map(lambda x1, x2, y: (x1, y))
    val_rgb = val.map(lambda x1, x2, y: (x1, y))

    model = models.cnn3d(stand_alone=True)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    filepath = './models/3dCNN ' + dataset + '.h5'
    # callbacks
    cp_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                  save_best_only=True, mode='min',
                                  save_weights_only=True)
    es_callback = EarlyStopping(monitor='val_accuracy', patience=5)
    lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4)
    callback_list = [es_callback, cp_callback, lr_callback]

    history = model.fit(train_rgb, epochs=settings.EPOCHS, batch_size=settings.BATCH_SIZE,
                        callbacks=callback_list, validation_data=val_rgb)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.title.set_text('3DCNN loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')

    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.title.set_text('3DCNN accuracy')
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    fig.savefig('./graphs/' + dataset + ' 3DCNN.png')
    # plt.show()


def train_ts_3dcnn(train, val,  dataset):

    model = models.ts_3dcnn()
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    filepath = './models/TS 3D-CNN ' + dataset + '.h5'
    # callbacks
    cp_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                  save_best_only=True, mode='min',
                                  save_weights_only=True)
    es_callback = EarlyStopping(monitor='val_accuracy', patience=5)
    lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4)
    callback_list = [es_callback, cp_callback, lr_callback]

    history = model.fit(train, epochs=settings.EPOCHS, batch_size=settings.BATCH_SIZE,
                        callbacks=callback_list, validation_data=val)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.title.set_text('TS with Optical Flow loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')

    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.title.set_text('TS with Optical Flow accuracy')
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    fig.savefig('./graphs/' + dataset + 'TS 3DCNN.png')

