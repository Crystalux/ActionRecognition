import process_data
import train_model
import test_models


def select_dataset(dataset: str) -> str:
    """
    :param dataset: choose from 'UCF11', 'HMDB51', 'UCF101'
    :return: dataset folder
    """
    dataset_path = './data/'
    dataset_folder = {
        'UCF11': 'UCF11_updated_mpg/',
        'HMDB51': 'hmdb51_org/',
        'UCF101': 'UCF-101/'
    }
    return dataset_path + dataset_folder[dataset]


def run_cnn3d(dataset: str):
    dataset_path = select_dataset(dataset)
    train, val, test = process_data.create_dataset(dataset_path, dataset)
    train_model.train_3dcnn(train, val, dataset)
    test_models.test_3dcnn(test, './models/3dCNN ' + dataset + '.h5')


def run_ts_cnn3d(dataset: str):
    dataset_path = select_dataset(dataset)
    train, val, test = process_data.create_dataset(dataset_path, dataset)
    train_model.train_ts_3dcnn(train, val, dataset)
    test_models.test_ts_3dcnn(test, './models/TS 3D-CNN ' + dataset + '.h5')


if __name__ == '__main__':
    run_ts_cnn3d('UCF11')

