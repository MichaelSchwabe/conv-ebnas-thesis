"""
Author: 
    Michael Schwabe
Version: 
    1.0
Date: 
    30.10.2021
Function: 
    handle genomes for an evolution Object. It initalized an SearchSpace for Neural Architeture Search with an evolutionary aproach.
    This Class generate, decode and mutate genomes for evolutionary aproach.

inspired, adopted and merge from ->
https://github.com/automl/auto-sklearn
https://github.com/automl/Auto-PyTorch
https://github.com/PaulPauls/Tensorflow-Neuroevolution
https://github.com/joeddav/devol
"""


import numpy as np
import random as rand
import math
from keras.models import Sequential

#TODO: evaluiere den breiten Einsatz des SeparableConv2D
# https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, Convolution2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
)

class GenomeHandler:
    '''
    Stellt einer Evolutionsumgebung einen Searchspace zur 
    Verfügung und kann die Architekturen bauen, mutieren 
    und aus der Kodierung entsprechend encoding and decoding 
    der Genomobjekte realisieren.
    Auch funktionen zur selektiven Extraktion sind vorhanden
    '''
    def __init__(self, max_conv_layers, max_dense_layers, max_filters,
                 max_dense_nodes, input_shape, n_classes,
                 batch_normalization=True, dropout=True, max_pooling=True,
                 optimizers=None, activations=None):
        '''
        Parameters
        --------------
        max_conv_layers: int
            Maximale Anzahl an Faltungsschichten
        max_dense_layers: int
            Maximale Anzahl an FullyCOnnectedLayer
        max_filters: int
            Maximale Filteranzahl in den Faltungsschichten
        max_dense_nodes: int
            Maximale Knotenanzahl in den FullyCOnnectedLayer
        input_shape: int
            Dimension des Inputs
        n_classes: int
            Wie viele Klassen sollen detektiert werden
        batch_normalization: boolean
            batch_normalization nutzen oder nicht (default True)
        dropout: boolean
            dropout nutzen oder nicht (default True)
        max_pooling: boolean
            max_pooling nutzen oder nicht (default True)
        optimizers: List (None = List from SearchSpace = Max)
            Liste von Strings die die optimierer begrenzen bzw. definieren die im Suchraum genutzt werden dürfen
        activations: List (None = List from SearchSpace = Max)
            Liste von Strings die die Aktievierungsfunktionen begrenzen bzw. definieren die im Suchraum genutzt werden dürfen
        '''
        
        if max_dense_layers < 1:
            raise ValueError(
                "choose minimal 1 dense Layer. The Last Layer must be an Dense-Layer with softmax activation"
            )
        if max_filters > 0:
            filter_range_max = int(math.log(max_filters, 2)) + 1
        else:
            filter_range_max = 0
        self.optimizer = optimizers or [
            #TODO: CHECK AND DEBUG
            'adam',
            'rmsprop',
            'adagrad',
            'adadelta',
            #NEW OPTIMIZER
            'sgd',
            'adamax',
            'nadam',
            'ftrl'
        ]
        self.activation = activations or [
            #TODO: CHECK AND DEBUG .. Wenn es zu fehlern führt kann es an den aktivierungsfunktionen liegen
            'relu',
            'sigmoid',
            #NEW ACTIVATIONS
            'softmax',
            'softplus',
            'softsign',
            'tanh',
            'selu',
            'elu',
            'exponential'
        ]
        self.convolutional_layer_shape = [
            "active",
            "num filters",
            "batch normalization",
            "activation",
            "dropout",
            "max pooling",
        ]
        self.dense_layer_shape = [
            "active",
            "num nodes",
            "batch normalization",
            "activation",
            "dropout",
        ]
        self.layer_params = {
            "active": [0, 1],
            "num filters": [2**i for i in range(3, filter_range_max)],
            "num nodes": [2**i for i in range(4, int(math.log(max_dense_nodes, 2)) + 1)],
            "batch normalization": [0, (1 if batch_normalization else 0)],
            "activation": list(range(len(self.activation))),
            "dropout": [(i if dropout else 0) for i in range(11)],
            "max pooling": list(range(3)) if max_pooling else 0,
        }

        self.convolution_layers = max_conv_layers
        self.convolution_layer_size = len(self.convolutional_layer_shape)
        self.dense_layers = max_dense_layers - 1  # this doesn't include the softmax layer, so -1
        self.dense_layer_size = len(self.dense_layer_shape)
        self.input_shape = input_shape
        self.n_classes = n_classes
        
    def show(self):
        '''
        StdOut Ausgabe der Klassenattribute
        ''' 
        print(self.optimizer, self.activation, self.convolutional_layer_shape, self.dense_layer_shape, 
              self.layer_params, self.convolution_layers, self.dense_layers, self.input_shape,
              self.n_classes)
    
    
    def generate(self):
        '''
        erzeugt eine Struktur mit WErtikkeiten bzw. parametrisiert das Genom, 
        für die Verwundung als Kandidat/Individuum in einer Population
               
        Returns
        --------------
        genome: List 
            eine Liste mit den elementen eines genomes
        '''
        #Array für die einzelnen Strukturen
        genome = []
        #SOlagne alle ConvLayer durch sind
        for i in range(self.convolution_layers):
            #für jeden Schlüssel
            for key in self.convolutional_layer_shape:
                #wähle aus jeder param Config ein zufälligen Wert aus der 
                #ParameterListe aus ... jeder Schlüssel des Dicts wird durchgegangen
                param = self.layer_params[key]
                #Zufallswahl ... ACHTUNG bei leerem EIntrag Exception
                #TODO: EXCEPTION HANDLING!
                genome.append(np.random.choice(param))
        #SOlagne alle DensLayer durch sind
        for i in range(self.dense_layers):
            #für jeden Schlüssel
            for key in self.dense_layer_shape:
                #wähle aus jeder param Config ein zufälligen Wert aus der 
                #ParameterListe aus ... jeder Schlüssel des DIcts wird durchgegangen
                param = self.layer_params[key]
                #Zufallswahl ... ACHTUNG bei leerem EIntrag Exception
                #TODO: EXCEPTION HANDLING!
                genome.append(np.random.choice(param))
        # Zufälliger Optimieren wird ausgewählt
        # TODO: OPTIMIERER erweitern
        genome.append(np.random.choice(list(range(len(self.optimizer)))))
        genome[0] = 1
        return genome
    
    # DEcode the Architectures
   
    def decode(self,genome):
        '''
        Kompiliert anhand des Genoms die Achitektur mit dem entsprechenden Framework, hier Keras 
        
        Parameters
        --------------
        genome: List
            genome aus der generate methode
        
        Returns
        --------------
        model: keras.model 
            object class from Keras
        '''
        
        #Keras ModelTyp initialisieren
        model = Sequential()
        
        #Steuerungvaiablen setzen
        dim = 0
        offset = 0 #dient der Steuerung des indexes des genomes
        
        if self.convolution_layers > 0:
            #wähle die kleinste Dimension,[min(X,X),X] 
            # -> # Bsp. [min(28,28),1] = 28, -> Bsp. [min(32,32),3] = 32
            dim = min(self.input_shape[:-1])  # keep track of smallest dimension
            #print("#####################################")
            #print(self.input_shape[:-1])
            #print(self.input_shape)
            #print("#####################################")
        input_layer = True
        
        #für alle ConvLayer laufe X
        for i in range(self.convolution_layers):
            # index 0 des Genomes .. wird später auf die via sliding window verschoben damit jede Schicht dabei ist
            if genome[offset]:
                # TODO: ersetze Convolution2d durch SeparableConv2D 
                # (schnellere Berechungen und finden sich auch in MobilNet wieder)
                # deklariere container
                convolution = None
                # nur bei der ersten Schicht nötig weitere Schichten haben keinen input mehr
                if input_layer:
                    convolution = SeparableConv2D(
                        genome[offset + 1], (3, 3),
                        padding='same',
                        input_shape=self.input_shape
                    )
                                  
                    input_layer = False
                # zählt nur für Schichten/Layer > 1 denn diese brauchen kein Inputshape mehr
                else:
                    convolution = SeparableConv2D(
                        genome[offset + 1], (3, 3),
                        padding='same'
                    )

                # füge die Schicht dem Modell hinzu
                model.add(convolution)
                
                #aktiviere Batchnormalisierung wenn nötig/vorgegeben
                if genome[offset + 2]:
                    model.add(BatchNormalization())
                    
                #initialisiere die Aktivierungsfunktion aus dem Genom
                model.add(Activation(self.activation[genome[offset + 3]]))
                #initialisiere Dropout aus dem Genom
                model.add(Dropout(float(genome[offset + 4] / 20.0)))
                #initialisiere max_pooling aus dem Genom
                max_pooling_type = genome[offset + 5]
                # Achtung ich muss überprüfen ob es groß genug ist ansonsten mache kein maxpooling
                if max_pooling_type == 1 and dim >= 5:
                    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
                    dim = int(math.ceil(dim / 2))
            # Setze den Index auf den nächsten Layer abschnitt im Genom
            # so werden verschiedene Layer aneinander gereit
            offset += self.convolution_layer_size
        # Wenn Input Layer = False dann füge eine Flatten Schicht hinzu bzw. Flatte den Output
        if not input_layer:
            model.add(Flatten())

        # für alle DenseLayer laufe X
        for i in range(self.dense_layers):
            if genome[offset]:
                # deklariere container
                dense = None
                 # nur bei der ersten Schicht nötig weitere Schichten haben keinen input mehr
                if input_layer:
                    dense = Dense(genome[offset + 1], input_shape=self.input_shape)
                    input_layer = False
                # zählt nur für Schichten/Layer > 1 denn diese brauchen kein Inputshape mehr
                else:
                    dense = Dense(genome[offset + 1])
                    
                # füge die Schicht dem Modell hinzu
                model.add(dense)
                
                # aktiviere Batchnormalisierung wenn nötig/vorgegeben
                if genome[offset + 2]:
                    model.add(BatchNormalization())
                    
                # initialisiere die Aktivierungsfunktion aus dem Genom
                model.add(Activation(self.activation[genome[offset + 3]]))
                
                # initialisiere Dropout aus dem Genom
                model.add(Dropout(float(genome[offset + 4] / 20.0)))
            # Setze den Index auf den nächsten Layer abschnitt im Genom
            # so werden verschiedene Layer aneinander gereit
            offset += self.dense_layer_size
        # füge am Ende eine FullyCOnnected Schicht hinzu letzter DenseLayer Softmax .. 
        # Achtung nur für den Versuchaufbau gut bei RNN,Transformern oder AutoEncodern 
        # muss hier eine Variation sein
        model.add(Dense(self.n_classes, activation='softmax'))
        
        # COmpilieren des Netzwerkes
        # TODO: Losses erweitern/randomisieren sowie die Metriken ausbauen 
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer[genome[offset]],
                      metrics=["accuracy"])
        # retrun the NN
        return model
    
    def convParam(self, i):
        '''
        Getter Methode um ein Klassenatribut zurückzu liefern .. speziell der ConvParameter
        
        Parameters
        --------------
        i: int
            Index des gewünschten Listen eintrages des Klassenattributes
        
        Returns
        --------------
        self.layer_params[self.convolutional_layer_shape[i]]: List
            Gibt eine Liste zurück aus einem Klassenattribut. Dafür ist ein Schlüssen nötig (dict) 
              
        '''
        # TODO: Debuggen
        # return self.layer_params[self.convolutional_layer_shape[i]]
        key = self.convolutional_layer_shape[i]
        return self.layer_params[key]
    
    def denseParam(self, i):
        '''
        Getter Methode um ein Klassenatribut zurückzu liefern .. speziell der DenseParameter
        
        Parameters
        --------------
        i: int
            Index des gewünschten Listen eintrages des Klassenattributes
        
        Returns
        --------------
        self.layer_params[self.dense_layer_shape[i]]: List
            Gibt eine Liste zurück aus einem Klassenattribut. Dafür ist ein Schlüssen nötig (dict) 
        '''
        # TODO: Debuggen
        # return self.layer_params[self.dense_layer_shape[i]]
        key = self.dense_layer_shape[i]
        return self.layer_params[key]
    
    
    def mutate(self, genome, num_mutations):
        '''
        Muttations Methode. Berechnet bzw. variert genome in Abhängigkeit zur Anzahl der Mutationen

        Parameters
        --------------
        genome: List
            zu mutierendes Genom

        i: int
            Anzahl der gewünschten Mutationen auf einem genome
        
        Returns
        --------------
        genome: List
            leifert ein mutiertes genome zurück
        '''
        # wähle eine Zufällige Anzahl an Mutationen
        num_mutations = np.random.choice(num_mutations)
        for i in range(num_mutations):
            # wähle zufällige Layer aus dem Genome
            index = np.random.choice(list(range(1, len(genome))))
            # wenn die Schicht noch in der range N-1 Layer ist
            if index < self.convolution_layer_size * self.convolution_layers:
                if genome[index - index % self.convolution_layer_size]:
                    range_index = index % self.convolution_layer_size
                    choice_range = self.convParam(range_index)
                    genome[index] = np.random.choice(choice_range)
                elif rand.uniform(0, 1) <= 0.01:  
                    # setze deactivate falg auf active
                    genome[index - index % self.convolution_layer_size] = 1
            # wenn der Index auf einen DenseLayer zeigt bzw. sicham ende befindet
            elif index != len(genome) - 1:
                offset = self.convolution_layer_size * self.convolution_layers
                new_index = (index - offset)
                # verschiebt die DenseSchichten
                present_index = new_index - new_index % self.dense_layer_size
                if genome[present_index + offset]:
                    range_index = new_index % self.dense_layer_size
                    choice_range = self.denseParam(range_index)
                    genome[index] = np.random.choice(choice_range)
                elif rand.uniform(0, 1) <= 0.01:
                    genome[present_index + offset] = 1
            else:
                # wähle einen anderen optimierer
                genome[index] = np.random.choice(list(range(len(self.optimizer))))
        return genome
    
    
    def genome_representation(self):
        '''
        Listen Repräsentation des Genomes. Hier wird die Grundlegende Architektur beschrieben.
        Es werden noch keien WErtigkeiten vergeben. 
        Das ist nur die strukturelle Repräsentation aus dem SearchSpace bzw. die Limitation aus diesem

        Returns
        --------------
        encoding: List
            leifert eine Genomstruktur .. keine Wertigkeiten .. Abhängig vom definierten Suchraum

        '''
        encoding = []
        for i in range(self.convolution_layers):
            for key in self.convolutional_layer_shape:
                encoding.append("Conv" + str(i) + " " + key)
        for i in range(self.dense_layers):
            for key in self.dense_layer_shape:
                encoding.append("Dense" + str(i) + " " + key)
        encoding.append("Optimizer")
        return encoding
        
    
    def is_compatible_genome(self, genome):
        '''
        Prüfen ob das ganze genome compatibel ist. Dazu zählt die Input und Output dimension
        genome:
            genome welches geprüft wird
        Retruen:
            True / False

        Parameters
        --------------
        genome: List
            zu prüfendes Genom

        Returns
        --------------
        True/False: boolean
            liefert je nach bedingung bzw. erfüllung True oder False
        '''
        expected_len = self.convolution_layers * self.convolution_layer_size \
            + self.dense_layers * self.dense_layer_size + 1
        if len(genome) != expected_len:
            return False
        ind = 0
        # Prüfe ConvLayer
        for i in range(self.convolution_layers):
            for j in range(self.convolution_layer_size):
                if genome[ind + j] not in self.convParam(j):
                    return False
            ind += self.convolution_layer_size
        # Prüfe DenseLayer
        for i in range(self.dense_layers):
            for j in range(self.dense_layer_size):
                if genome[ind + j] not in self.denseParam(j):
                    return False
            ind += self.dense_layer_size
        if genome[ind] not in range(len(self.optimizer)):
            return False
        return True
    
    
    def best_genome(self, csv_path, metric="accuracy", include_metrics=True):
        '''
        Extrahiert das beste genom aus einem csv_path. Muss unbedingt der Stuktur des Algos entsprechen!!!

        Parameters
        --------------
        csv_path: String
            Pfad zum Genomregister
        metric: String
            accuracy oder loss
        include_metrics: boolean
            Ture oder False

        Returns
        --------------
        genome: List
            liefert das beste genom
        '''
        #min = Loss, max = acc
        best = max if metric is "accuracy" else min
        col = -1 if metric is "accuracy" else -2
        #nutze numpy um die genom config zu lesen
        data = np.genfromtxt(csv_path, delimiter=",")
        # zeilenweise .. best = max oder min
        row = list(data[:, col]).index(best(data[:, col]))
        
        genome = list(map(int, data[row, :-2]))
        if include_metrics:
            genome += list(data[row, -2:])
        return genome
    
    
    def decode_best(self, csv_path, metric="accuracy"):
        '''
        decodierung des besten genomes in ein Keras Model

        Parameters
        --------------
        csv_path: String
            Pfad zum Genomregister
        metric: String
            accuracy oder loss
        include_metrics: boolean
            Ture oder False

        Returns
        --------------
        model: keras.model
            decodiert ein genome und gibt ein Keras Model zurück
        '''
        #dekodiere/definiere in einem neuen Genom den besten Kandidaten
        return self.decode(self.best_genome(csv_path, metric, False))