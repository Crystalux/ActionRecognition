import models


def test_3dcnn_lstm(test_ds, model_weights):
    model = models.CNN3d_LSTM()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(model_weights)
    loss, acc = model.evaluate(test_ds, verbose=2)
    print('Model accuracy: {:5.2f}%'.format(100 * acc))
