import models
import settings
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train_convlstm(train_ds, val_ds):
    opt = keras.optimizers.SGD(lr=0.001)
    model = models.conv_lstm()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Define the check point
    filepath = './models/conv_lstm-{epoch:02d}-{loss:.4f}.h5'
    cp_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                  save_best_only=True, mode='min')
    callback_list = [cp_callback]
    model.fit(train_ds, epochs=settings.EPOCHS, batch_size=settings.BATCH_SIZE,
              callbacks=callback_list, validation_data=val_ds)
