import process_data
import train_model
import process_data
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


def run_cnn3d_lstm(dataset: str):
    dataset_path = select_dataset(dataset)
    train_ds, val_ds, test_ds = process_data.create_dataset(dataset_path, dataset)
    train_model.train_3dcnn_lstm(train_ds, val_ds, dataset)
    test_models.test_3dcnn_lstm(test_ds, './models/3dCNN_LSTM ' + dataset + '.h5')


if __name__ == '__main__':
    run_cnn3d_lstm('UCF11')

