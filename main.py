import process_data
import train_model


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


def main(dataset: str):
    dataset_path = select_dataset(dataset)
    train_ds, val_ds, test_ds = process_data.create_dataset(dataset_path, dataset)
    # train_model.train_resnet_lstm(train_ds, val_ds)
    # takes 2.5 hours
    # train_model.train_convlstm(train_ds, val_ds)

    train_model.train_3dcnn_lstm(train_ds, val_ds)


if __name__ == '__main__':
    main('UCF11')
