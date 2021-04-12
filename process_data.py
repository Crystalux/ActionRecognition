import os
import re
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import settings

settings.init()


def create_path_pd(dataset_path):
    categories = os.listdir(dataset_path)
    path_list = []
    for category in categories:
        for root, dirs, files in os.walk(dataset_path + '/' + category):
            for file in files:
                if file.endswith('.mpg'):
                    path = os.path.join(root, file)
                    path = re.compile(r"[\/]").split(path)
                    # join the path using the correct slash symbol:
                    path = os.path.join(*path).replace("\\", "/")
                    path_list.append({'path': path, 'label': category})
    df = pd.DataFrame(path_list)
    return categories, df


def remove_video(path, n_frames=settings.N_FRAMES):
    vidcap = cv2.VideoCapture(path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vidcap.release()
    return total_frames < n_frames


def video_to_array(path):
    vid_frames = []
    vidcap = cv2.VideoCapture(path.numpy().decode('utf-8'))
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list = np.linspace(0, total_frames - 1, settings.N_FRAMES, dtype=np.int16)

    if total_frames < settings.N_FRAMES:
        return None, None

    for fn in range(total_frames):
        ret, frame = vidcap.read()
        if not ret:
            continue
        if fn in frame_list:
            # frames fixed to 224x244, ignoring aspect ratio
            frame = cv2.resize(frame, (settings.NUM_SIZE, settings.NUM_SIZE))

            # change frame to RGB and append to frames
            if settings.NUM_CHANNELS == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = frame[:, :, np.newaxis]

            arr = np.asarray(frame)
            norm_arr = (arr / 255).astype(np.float32)
            vid_frames.append(norm_arr)
    vidcap.release()

    vid_frames = np.array(vid_frames)
    vid_frames = tf.convert_to_tensor(vid_frames, dtype=tf.float32)
    vid_frames.set_shape([settings.N_FRAMES, settings.NUM_SIZE, settings.NUM_SIZE, settings.NUM_CHANNELS])
    return vid_frames


def mapping_func(path, label):
    [vid_frames, ] = tf.py_function(func=video_to_array, inp=[path], Tout=[tf.float32])
    vid_frames.set_shape(settings.TENSOR_SHAPE)
    return vid_frames, label


# def mapping_write_fn(path, label):
#     return tf.io.serialize_tensor(path), label


def create_dataset(dataset_path, data_name):
    categories, df = create_path_pd(dataset_path)
    settings.NUM_CLASSES = len(categories)

    print('Removing files due to insufficient frames\n')
    for path in tqdm(df['path']):
        if remove_video(path):
            df.drop(df.index[df['path'] == path], inplace=True)
    print('\nRemaining files: ', len(df.index))

    print('Creating tf.data.Dataset')
    file_paths = df['path'].to_numpy()
    labels = df['label'].values
    label_ecoder = LabelEncoder()

    transformed_label = label_ecoder.fit_transform(labels)
    label_cat = keras.utils.to_categorical(transformed_label, num_classes=settings.NUM_CLASSES)

    X_train, X_test, y_train, y_test = train_test_split(file_paths, label_cat, test_size=0.2, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=True)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.map(mapping_func).batch(settings.BATCH_SIZE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.map(mapping_func).batch(settings.BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.map(mapping_func).batch(settings.BATCH_SIZE)

    return train_ds, val_ds, test_ds



