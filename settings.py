import tensorflow as tf


def init():
    global NUM_SIZE
    global NUM_CHANNELS
    global N_FRAMES
    global NUM_CLASSES
    global BATCH_SIZE
    global EPOCHS
    global TENSOR_SHAPE
    global BUFFER_SIZE

    NUM_SIZE = 64
    NUM_CHANNELS = 3
    N_FRAMES = 10
    BATCH_SIZE = 2
    EPOCHS = 50
    TENSOR_SHAPE = tf.TensorShape([N_FRAMES, NUM_SIZE, NUM_SIZE, NUM_CHANNELS])
    NUM_CLASSES = None
    BUFFER_SIZE = 6000


