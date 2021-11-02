from __future__ import print_function
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
import numpy as np
from keras import backend as K
from evolution import Evolution
from genome_handler import GenomeHandler
import tensorflow as tf
#import mlflow.keras
#import mlflow
#import mlflow.tensorflow
#mlflow.tensorflow.autolog() 
#mlflow.keras.autolog()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

K.set_image_data_format("channels_last")

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],x_train.shape[3]).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3]).astype('float32') / 255
# nCLasses
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#y_train.shape
dataset = ((x_train, y_train), (x_test, y_test))

genome_handler = GenomeHandler(max_conv_layers=20, 
                               max_dense_layers=8, # includes final dense layer
                               max_filters=2048,
                               max_dense_nodes=1024,
                               input_shape=x_train.shape[1:],
                               n_classes=10)


evo = Evolution(genome_handler, data_path="log/evo_cifar10_gen40_pop5_e5.csv")
model = evo.run(dataset=dataset,
                  num_generations=80,
                  pop_size=5,
                  epochs=5)
print(model.summary())