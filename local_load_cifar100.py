import pickle
import numpy as np
from keras.utils.np_utils import to_categorical

def unpickle(file):
    with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar100(path="data/"):
    

    meta = unpickle(path+'meta')
    train = unpickle(path+'train')
    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
    filenames = [t.decode('utf8') for t in train[b'filenames']]
    fine_labels = train[b'fine_labels']

    test = unpickle(path+'test')
    test_filenames = [t.decode('utf8') for t in test[b'filenames']]
    test_fine_labels = test[b'fine_labels']
    
    train_data = train[b'data']
    test_data = test[b'data']
    
    train_images = list()
    for d in train_data:
        image = np.zeros((32,32,3), dtype=np.uint8)
        image[...,0] = np.reshape(d[:1024], (32,32)) # Red channel
        image[...,1] = np.reshape(d[1024:2048], (32,32)) # Green channel
        image[...,2] = np.reshape(d[2048:], (32,32)) # Blue channel
        train_images.append(image)

    test_images = list()
    for d in test_data:
        image = np.zeros((32,32,3), dtype=np.uint8)
        image[...,0] = np.reshape(d[:1024], (32,32)) # Red channel
        image[...,1] = np.reshape(d[1024:2048], (32,32)) # Green channel
        image[...,2] = np.reshape(d[2048:], (32,32)) # Blue channel
        test_images.append(image)


    x_train = np.array(train_images)
    y_train = np.array(fine_labels)
    y_train = np.reshape(y_train,(50000,1))
    x_test = np.array(test_images)
    y_test = np.array(test_fine_labels)
    y_test = np.reshape(y_test,(10000,1))

    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)
    

    return (x_train, y_train), (x_test, y_test)