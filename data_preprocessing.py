import os
import numpy as np
from keras.datasets import mnist


def load_preprocess_data(num_classes=10):
    """
    Load and preprocess the MNIST dataset with memoization.

    Returns:
        x_train, y_train, x_test, y_test: preprocessed training and test data
    """
    data_file_name = f'data_{num_classes}.npz'

    if os.path.exists(data_file_name):
        # Load preprocessed data from the file
        data = np.load(data_file_name)
        x_train, y_train, x_test, y_test = (
            data['x_train'], data['y_train'], data['x_test'], data['y_test'])
    else:
        # Load and preprocess the data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        y_train = np.eye(num_classes)[y_train]
        y_test = np.eye(num_classes)[y_test]

        # Save the preprocessed data to a file
        np.savez(data_file_name, x_train=x_train,
                 y_train=y_train, x_test=x_test, y_test=y_test)

    return x_train, y_train, x_test, y_test
