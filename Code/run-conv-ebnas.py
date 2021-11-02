from __future__ import print_function
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
import numpy as np
from keras import backend as K
from evolution import Evolution
from genome_handler import GenomeHandler
import tensorflow as tf


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


K.set_image_data_format("channels_last")

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
dataset = ((x_train, y_train), (x_test, y_test))

# Definition des Searchspace inklusive der Definition der Klassen für den 
# Output-Layer sowie die Dimension des Inputs
genome_handler = GenomeHandler(max_conv_layers=10, 
                               max_dense_layers=4, #inklusive des finalen DenseLayer
                               max_filters=1024,
                               max_dense_nodes=512,
                               input_shape=x_train.shape[1:],
                               n_classes=10)

# Erstellen des genetischen Programms bzw. der Umgebung für die Evaluation von
# verschiednene Individuen 
evo = Evolution(genome_handler, data_path="log.csv")
model = evo.run(dataset=dataset,
                  num_generations=20,
                  pop_size=20,
                  epochs=5)
print(model.summary())