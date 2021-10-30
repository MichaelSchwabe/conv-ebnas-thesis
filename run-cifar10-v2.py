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

# **Prepare dataset**
# This problem uses mnist, a handwritten digit classification problem used
# for many introductory deep learning examples. Here, we load the data and
# prepare it for use by the GPU. We also do a one-hot encoding of the labels.

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

# **Prepare the genome configuration**
# The `GenomeHandler` class handles the constraints that are imposed upon
# models in a particular genetic program. See `genome-handler.py`
# for more information.

genome_handler = GenomeHandler(max_conv_layers=10, 
                               max_dense_layers=4, # includes final dense layer
                               max_filters=1024,
                               max_dense_nodes=512,
                               input_shape=x_train.shape[1:],
                               n_classes=10)
# **Create and run the genetic program**
# The next, and final, step is create a `DEvol` and run it. Here we specify
# a few settings pertaining to the genetic program. The program
# will save each genome's encoding, as well as the model's loss and
# accuracy, in a `.csv` file printed at the beginning of program.
# The best model is returned decoded and with `epochs` training done.

evo = Evolution(genome_handler, data_path="log/evo_cifar10_gen20_pop20_e5.csv")
model = evo.run(dataset=dataset,
                  num_generations=20,
                  pop_size=20,
                  epochs=5)
print(model.summary())