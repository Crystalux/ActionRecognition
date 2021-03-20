import os
import cv2
import numpy as np


def get_frames(filename, n_frames=1):
    """
    :param filename: path and name of video file
    :param n_frames: number of frames to extract
    :return: list of frames extracted and total number of frames the original video had
    """
    frames = []
    vidcap = cv2.VideoCapture(filename)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list = np.linspace(0, total_frames - 1, n_frames, dtype=np.int16)

    for fn in range(total_frames):
        ret, frame = vidcap.read()
        if not ret:
            continue
        if fn in frame_list:
            # change frame to RGB and append to frames
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frames fixed to 224x244, ignoring aspect ratio
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    vidcap.release()
    return frames, total_frames


def store_frames(frames, store_path):
    """
    :param frames: list of cv2 frames
    :param store_path: Path to store frames in
    :return: None
    """
    for idx, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_path = os.path.join(store_path, 'frame{:02d}.jpg'.format(idx)).replace('\\', '/')
        cv2.imwrite(image_path, frame)


class Preprocess:
    def __init__(self, dataset_folder):

        self.data_path = './data/'
        self.dataset_path = self.data_path + dataset_folder + '/'
        self.jpg_path = self.data_path + dataset_folder + '_jpg/'
        # List of categories
        self.list_categories = os.listdir(self.dataset_path)

        # Create JPG folder
        try:
            os.makedirs(self.jpg_path)
            print('Folder {} created.'.format(self.jpg_path))
            self.do_preprocess = True
        except:
            print('Folder {} exists.'.format(self.jpg_path))
            self.do_preprocess = False

    def should_do_preprocess(self):
        return self.do_preprocess

    def preprocess(self):
        # print number of subfolders in each category
        for cat in self.list_categories:
            print('Category: ', cat)
            category_path = os.path.join(self.dataset_path, cat).replace('\\', '/')
            list_of_sub = os.listdir(category_path)
            print('number of subfolders: ', len(list_of_sub))
            print('=' * 50)

        # Save each frame as a jpg
        extension = '.mpg'
        n_frames = 16
        for root, dirs, files in os.walk(self.dataset_path, topdown=False):
            for name in files:
                if name[-4:] != extension:
                    continue
                video_path = os.path.join(root, name).replace('\\', '/')
                frames, total_frame = get_frames(video_path, n_frames=n_frames)
                store_path = video_path.replace(self.dataset_path, self.jpg_path)
                store_path = store_path.replace(extension, '')

                os.makedirs(store_path, exist_ok=True)
                store_frames(frames, store_path)
            print('=' * 50)

        # Create numpy array

