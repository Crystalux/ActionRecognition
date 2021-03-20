import tensorflow as tf
from preprocess.preprocess import Preprocess
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    UCF11 = Preprocess('UCF11_updated_mpg')
    if UCF11.should_do_preprocess():
        UCF11.preprocess()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
