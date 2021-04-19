import models


def test_3dcnn(test, model_weights):
    test_rgb = test.map(lambda x1, x2, y: (x1, y))
    model = models.cnn3d()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(model_weights)
    loss, acc = model.evaluate(test_rgb, verbose=2)
    print('Model accuracy: {:5.2f}%'.format(100 * acc))
    print('Model loss: {}'.format(loss))


def test_ts_3dcnn(test, model_weights):
    test_rgb = test.map(lambda x1, x2, y: ([x1, x2], y))
    model = models.cnn3d()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(model_weights)
    loss, acc = model.evaluate(test_rgb, verbose=2)
    print('Model accuracy: {:5.2f}%'.format(100 * acc))
    print('Model loss: {}'.format(loss))

