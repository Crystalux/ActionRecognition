import models
import settings
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt


def train_3dcnn_lstm(train_ds, val_ds, dataset):
    model = models.CNN3d_LSTM()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    filepath = './models/3dCNN_LSTM ' + dataset + '.h5'
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
    ax1.legend(['Train', 'Validation'], loc='upper left')

    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.title.set_text('3D CNN-LSTM accuracy')
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    fig.savefig('./graphs/' + dataset + ' 3DCNN-LSTM.png')
    plt.show()

