import os
import numpy as np


class Dataset(object):

    def __init__(self, input_dim):

        self.shape = [input_dim, 1]
        self.input_dim = input_dim
        self.data_x, self.data_y, self.test_x, self.test_y = self.load_data()

    def load_data(self):

        data_dir = os.path.join("./data")
        fd = open(os.path.join(data_dir, 'X_train_by_random_4d.npy'))
        loaded = np.fromfile(file=fd, dtype=np.float)
        X_train = loaded.reshape((-1, self.input_dim))

        fd = open(os.path.join(data_dir, 'Y_train_by_random_4d.npy'))
        loaded = np.fromfile(file=fd, dtype=np.float)
        Y_train = loaded.reshape((-1, 1))

        fd = open(os.path.join(data_dir, 'X_test_4d.npy'))
        loaded = np.fromfile(file=fd, dtype=np.float)
        X_test = loaded.reshape((-1, self.input_dim))

        fd = open(os.path.join(data_dir, 'Y_test_4d.npy'))
        loaded = np.fromfile(file=fd, dtype=np.float)
        Y_test = loaded.reshape((-1, 1))

        return X_train, Y_train, X_test, Y_test


    def getNext_batch(self, iter_num=0, batch_size=64):

        ro_num = len(self.data) / batch_size - 1

        if iter_num % ro_num == 0:

            length = len(self.data_x)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.data_x = np.array(self.data_x)
            self.data_x = self.data_x[perm]
            self.data_y = np.array(self.data_y)
            self.data_y = self.data_y[perm]

        return self.data_x[int(iter_num % ro_num) * batch_size: int(iter_num% ro_num + 1) * batch_size] \
            , self.data_y[int(iter_num % ro_num) * batch_size: int(iter_num%ro_num + 1) * batch_size]


def sample_output(batch_size, min, max):
    label_vector = np.random.rand(batch_size, 1) * (max-min) + min
    return label_vector


def save_samples(sample_feature, sample_output):
    result = np.concatenate([sample_feature, sample_output], axis=1)
    np.save('x.npy', result, allow_pickle=False)
