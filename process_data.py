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
                if file.endswith('.mpg') or file.endswith('.avi'):
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
    # Read the first frame
    vidcap.set(1, 0)
    ret, _ = vidcap.read()
    # if the first frames was unable to be read, remove file.
    if not ret:
        return True
    vidcap.release()
    return total_frames < n_frames


def rgb_and_optical(path):
    rgb_frames = []
    opt_frames = []
    vidcap = cv2.VideoCapture(path.numpy().decode('utf-8'))
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_list = np.linspace(0, total_frames - 1, settings.N_FRAMES, dtype=np.int16)
    frame_step = total_frames // settings.N_FRAMES

    for i in range(settings.N_FRAMES):
        vidcap.set(1, i*frame_step)
        ret, frame1 = vidcap.read()
        if not ret:
            print('\n', path.numpy().decode('utf-8'))
        if i == 0:
            vidcap.set(1, i*frame_step)
        else:
            vidcap.set(1, i*frame_step - 1)
        ret, frame2 = vidcap.read()
        if not ret:
            print('\n', path.numpy().decode('utf-8'))

        frame1 = cv2.resize(frame1, (settings.NUM_SIZE, settings.NUM_SIZE))
        frame2 = cv2.resize(frame2, (settings.NUM_SIZE, settings.NUM_SIZE))

        # change frame to RGB and append to frames
        if settings.NUM_CHANNELS == 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        else:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame1 = frame1[:, :, np.newaxis]

        arr = np.asarray(frame1)
        norm_arr = (arr / 255).astype(np.float32)
        rgb_frames.append(norm_arr)

        # change frame to otical flow and append to frames
        # Create an image filled with zero intensities with the same dimension as the frame1
        mask = np.zeros_like(frame1)
        # Sets image saturation to maximum
        mask[..., 1] = 255

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # Create dencse optical flow
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # calculate magnitude and angle of the flow
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # set value according to angle and magnitude of flow
        mask[..., 0] = ang * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # convert hsv to rgb
        opt = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)

        arr = np.asarray(opt)
        norm_arr = (arr / 255).astype(np.float32)
        opt_frames.append(norm_arr)

    vidcap.release()

    rgb_frames = np.array(rgb_frames)
    rgb_frames = tf.convert_to_tensor(rgb_frames, dtype=tf.float32)
    rgb_frames.set_shape([settings.N_FRAMES, settings.NUM_SIZE, settings.NUM_SIZE, settings.NUM_CHANNELS])

    opt_frames = np.array(opt_frames)
    opt_frames = tf.convert_to_tensor(opt_frames, dtype=tf.float32)
    opt_frames.set_shape([settings.N_FRAMES, settings.NUM_SIZE, settings.NUM_SIZE, settings.NUM_CHANNELS])

    return rgb_frames, opt_frames


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


def create_optical_flow(path):
    path_str = path.numpy().decode('utf-8')
    temp_path = './data/temp/' + os.path.basename(path_str) + '.avi'
    vidcap = cv2.VideoCapture(path_str)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
    ret, first_frame = vidcap.read()
    width = int(vidcap.get(3))  # float `width`
    height = int(vidcap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(first_frame)
    # Sets image saturation to maximum
    mask[..., 1] = 255

    while (vidcap.isOpened()):
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = vidcap.read()
        if not ret:
            break
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        out.write(rgb)
        # Updates previous frame
        prev_gray = gray
        # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # The following frees up resources and closes all windows
    vidcap.release()
    out.release()

    return tf.convert_to_tensor(temp_path)


def clean_optical_flow(temp_path):
    os.remove(temp_path.numpy().decode('utf-8'))


def mapping_func(path, label):
    [vid_frames, ] = tf.py_function(func=video_to_array, inp=[path], Tout=[tf.float32])
    vid_frames.set_shape(settings.TENSOR_SHAPE)
    return vid_frames, label


def mapping_opt(path, label):
    [temp_path, ] = tf.py_function(func=create_optical_flow, inp=[path], Tout=[tf.string])
    [vid_frames, ] = tf.py_function(func=video_to_array, inp=[temp_path], Tout=[tf.float32])
    tf.py_function(func=clean_optical_flow, inp=[temp_path], Tout=[])
    vid_frames.set_shape(settings.TENSOR_SHAPE)
    return vid_frames, label


def map_rgb_opt(path, label):
    [rgb_frames, opt_frames, ] = tf.py_function(func=rgb_and_optical, inp=[path], Tout=[tf.float32, tf.float32])
    rgb_frames.set_shape(settings.TENSOR_SHAPE)
    opt_frames.set_shape(settings.TENSOR_SHAPE)
    return {'rgb': rgb_frames, 'optical': opt_frames}, label

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

    X_train, X_test, y_train, y_test = train_test_split(file_paths, label_cat, test_size=0.2,
                                                        shuffle=True, random_state=128)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                                      shuffle=True, random_state=42)

    with tf.device('/cpu:0'):
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train = train_ds.map(map_rgb_opt).batch(settings.BATCH_SIZE).prefetch(1)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val = val_ds.map(map_rgb_opt).batch(settings.BATCH_SIZE).prefetch(1)

        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test = test_ds.map(map_rgb_opt).batch(settings.BATCH_SIZE).prefetch(1)

    return train, val, test


