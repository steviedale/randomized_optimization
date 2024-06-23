from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from keras.datasets import mnist
import numpy as np
import cv2

def get_mnist_data():
    (_, _), (test_X, test_y) = mnist.load_data()
    X = test_X
    y = test_y

    # resize images to 14x14
    new_size = 14
    new_X = np.zeros((X.shape[0], new_size, new_size))
    for i in range(X.shape[0]):
        new_X[i] = cv2.resize(X[i], (new_size, new_size))
    X = new_X

    # flatten 14*14 images to a 784 vector for each image
    X = X.reshape(X.shape[0], -1)

    y_one_hot = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()

    # reduce dataset by 1/10
    skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
    _, test_index = list(skf.split(X, y))[0]
    X = X[test_index]
    y = y[test_index]
    y_one_hot = y_one_hot[test_index]

    return X, y, y_one_hot