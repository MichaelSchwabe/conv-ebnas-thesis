"""
Author: 
    Michael Schwabe
Version: 
    1.0
Date: 
    30.10.2021
Function: 
    Evolution an protected _Populaiton class. 
    Finalize the SearchSaace, SearchStradegy and Performance Estimation Strategy from NAS

inspired, adopted and merge from ->
https://github.com/automl/auto-sklearn
https://github.com/automl/Auto-PyTorch
https://github.com/PaulPauls/Tensorflow-Neuroevolution
https://github.com/joeddav/devol
"""


from __future__ import print_function
import random as rand
import csv
import operator
import gc
import os
from datetime import datetime
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.models import load_model
import keras.backend as K
from sklearn.metrics import log_loss
import numpy as np
from datetime import datetime
from keras_visualizer import visualizer 

if K.backend() == 'tensorflow':
    import tensorflow as tf

__all__ = ['Evolution']

#Konstanten
METRIC_OPS = [operator.__lt__, operator.__gt__]
METRIC_OBJECTIVES = [min, max]

#Evolution
class Evolution:
    '''
    finalisiert bzw. definiert den Suchraum abschließend, stellt die Suchstrategie und die Performance Estimation Stradegy 
    sowie eine Evolutionsumgebung und eine Populationsobjekt zur Verfügung.
    '''

    def __init__(self, genome_handler, data_path=""):
        '''
        Initialisiert ein Evolutionsobjekt .. trainiert und evaluiert nach dem prinzip der 
        evolutionären Hyperparametersuche bzw. genetischen Suche
        
        Parameters
        --------------
        genome_handler: GenomeHandler
            Definiert die eigentlichen Limitationen bzw. legt den Suchraum/SearchSpace fest.
            !Achtung hier gibt es bereits eine Verzerrung!
        data_path: String
            Speicherpfad zu den Encodings sowie den Metriken
        '''
        self.genome_handler = genome_handler
        self.datafile = data_path or (datetime.now().ctime() + '.csv')
        self._bssf = -1

        if os.path.isfile(data_path) and os.stat(data_path).st_size > 1:
            raise ValueError(('Datei %s bereits vorhanden. Datei Löschen oder Pfad ändern!' % data_path))

        print("Genome encoding und Metriken werden in dem folgenden Pfad abgelegt -> ", self.datafile, "\n")
        with open(self.datafile, 'a+') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            metric_cols = ["Val Loss", "Val Accuracy"]
            genome = genome_handler.genome_representation() + metric_cols
            writer.writerow(genome)
            
    def set_objective(self, metric):
        '''
        Setzt das Ziel fest (min=loss .. max=acc)
        
        Parameters
        --------------
        metric: String
            acc = max und loss = min
        '''
        if metric == 'acc':
            metric = 'accuracy'
        if metric not in ['loss', 'accuracy']:
            raise ValueError(('Metrik {} wird nicht untersützt - "accuracy" oder "loss"').format(metric))
        self._metric = metric

        #uebertrag nach was optimiert werden soll, min(loss) oder max(acc)
        self._objective = "max" if self._metric == "accuracy" else "min"
        self._metric_index = 1 if self._metric == 'loss' else -1
        self._metric_op = METRIC_OPS[self._objective == 'max']
        self._metric_objective = METRIC_OBJECTIVES[self._objective == 'max']
        
    def run(self, dataset, num_generations, pop_size, epochs, fitness=None,
        metric='accuracy'):
        '''
        Beginnt den Evolutionären Prozess zum finden einer geeigneten Architektur

        Parameters
        --------------
        dataset : (List,List) (List,List)
            ähnlich --> (train_data,train_labels), (validation_data, validation_labels) 
        num_generations: int
            Anzahl der generationen die gesucht werden soll
        pop_size: int
            Anzahl der initialen Populationsgröße
        epochs: int
            Anzahl der Epochen die für jedes Modell aufgewendet werden soll
        fitness: None, Float
            Scoring-Wert wird auf ein Numpy-Array aufgerufen, das eine min/max-skalierte Version der ausgewerteten
            Modellmetriken ist, also eine reelle Zahl einschließlich 0 akzeptieren sollte. 
            Bleibt es bei der Voreinstellung, werden nur die min/max skalierten Werte verwendet.
        metric: String
            Entweder "accuracy" oder "loss" sind als Metrik wählbar. Entscheidet nach was optimiert werden soll während der Suche

        Returns:
        --------------
        model: keras.model
            Das beste Modell wird zurückgegeben
        '''
        self.set_objective(metric)

        # If no validation data is given set it to None
        if len(dataset) == 2:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
            self.x_val = None
            self.y_val = None
        else:
            (self.x_train, self.y_train), (self.x_test, self.y_test), (self.x_val, self.y_val) = dataset

        # generate and evaluate initial population
        members = self._generate_random_population(pop_size)
        pop = self._evaluate_population(members,
                                        epochs,
                                        fitness,
                                        0,
                                        num_generations)

        # evolve
        for gen in range(1, num_generations):
            members = self._reproduce(pop, gen)
            pop = self._evaluate_population(members,
                                            epochs,
                                            fitness,
                                            gen,
                                            num_generations)

        return load_model('best-model.h5')

    def _reproduce(self, pop, gen):
        '''
        Reproduziert eine Population für die nächste Generation
        
        Parameters
        --------------
        pop: _Population
            Populationsobjekt der abstrakten Klasse _Population
        gen: int
            Zahl der aktuellen Generation
        
        Returns
        --------------
        members: List
            Liste der mutierten Mitglieder der nächsten Population
        '''
        members = []
        # TODO: Optimieren um Innovationen länger leben zu lassen aber auch um neue Innovationen zu ermöglichen
        # 95% der Population soll aus Kreuzung entstehen
        for _ in range(int(len(pop) * 0.95)):
            #zufällige Auswahl aus der Population mit pop.select()
            members.append(self._crossover(pop.select(), pop.select()))

        # Das Beste Modell überlebt immer 
        members += pop.get_best(len(pop) - int(len(pop) * 0.95))

        # Einleitung der zufälligen Mutation
        for imem, mem in enumerate(members):
            members[imem] = self._mutate(mem, gen)
        return members
    
   
    def _evaluate(self, genome, epochs):
        '''
        Parameters
        --------------
        genome: List
            genome Listen Objekt
        epochs: int
            Anzahl der epochen die ein Modell evaluiert werden soll
        
        Parameters
        --------------
        model: keras.model
            Model Obejkr des Keras Frameworks
        loss: float
            Metrik
        accuracy: float
            Metrik
        '''
        #decodierung des genomes
        model = self.genome_handler.decode(genome)
        loss, accuracy = None, None
        #paramerter Array .. mit unterandem den CallBacks und deren objectives
        fit_params = {
            'x': self.x_train,
            'y': self.y_train,
            'validation_split': 0.1,
            'epochs': epochs,
            'verbose': 1,
            'callbacks': [
                # TODO: Gebe Innovationen eine Chance patience=1 .. vielleicht 2 oder 3
                # entscheidet das nach der ersten Epoche die keine Verbesserung mit sich bringt abgebrochen wird ...
                # EarlyStopping(monitor='val_loss', patience=2, verbose=1)
                #### 
                # NEW
                EarlyStopping(monitor='val_loss', patience=1, verbose=1),
                TensorBoard(log_dir='./tfboardlogs', write_graph=True, write_images=False)
                #, histogram_freq=1, write_graph=True, write_images=True)
                ####
                #TODO: reduce_on_plateu .. Adaptiv Learningrate
                #ReduceLROnPlateau(
                #    monitor="val_loss",
                #    factor=0.1,
                #    patience=10,
                #    verbose=0,
                #    mode="auto",
                #    min_delta=0.0001,
                #    cooldown=0,
                #    min_lr=0,
                #    **kwargs
                #)
            ]
        }

        if self.x_val is not None:
            fit_params['validation_data'] = (self.x_val, self.y_val)
        #TryCatch Block für die Evaluaiton bzw. das Trainieren
        try:
            #fitte das Modell
            model.fit(**fit_params)

            #evalusationsmethoden aufruf um loss und acc zu holen
            #methode kommt aus dem Kearas/tf Framework
            loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
        except Exception as e:
            #aufruf wie mit einem defekten Modell umgegangen wird
            loss, accuracy = self._handle_broken_model(model, e)
        #Klassenmethode zum speicher der stats usw. csv SPeicherung, save best model usw.
        self._record_stats(model, genome, loss, accuracy)
        

        # zeitliche Komponenten bzw. speichern des Tiomestamps 
        dt = datetime.now()
        dt = datetime.timestamp(dt)
        #visualizer(model, filename=('log/modelgraph.'+str(dt)), format='png', view=False)
        lines = ['Loss: '+str(loss), 'Accuracy: '+str(accuracy)]
        #speichern der csv datei
        with open(str('log/model_'+str(dt)+'.txt'), 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
        
        # TODO: h5 Format wird in zukunft schwierig .. wechsel zu ONNX oder ein anderes Format
        #modell mit timestamp speichern zur Nachweisführung
        model.save(str('log/model_'+str(dt)+'.h5'))
        return model, loss, accuracy

    def _record_stats(self, model, genome, loss, accuracy):
        '''
        # TODO: h5 Format wird in zukunft schwierig .. wechsel zu ONNX oder ein anderes Format
        Methode dient dem sauberen ablegen der Metriken aber auch der Modelle in einem h5 Format
       
        Parameters
        --------------
        model: keras.model
            Modell Objekt welches berarbeitet werden soll
        genome: List
            COdiertes genome welches als Nachweis den Metriken beigelegt wird
        loss: float
            Verlustfunktion bzw. deren Validierungswert
        accuracy: float
            Genauigkeit bzw. deren Validierungswert

        Returns
        --------------
        members: List
            Liste der mutierten Mitglieder der nächsten Population
        '''
        #CSV write
        with open(self.datafile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = list(genome) + [loss, accuracy]
            writer.writerow(row)
        # Ziel check loss oder acc
        met = loss if self._metric == 'loss' else accuracy
        if (self._bssf is -1 or
                self._metric_op(met, self._bssf) and
                accuracy is not 0):
            try:
                # altes modell wenn möglich löschen
                os.remove('best-model.h5')
            except OSError:
                pass
            # beste Metreik in Klassenvariable ablegen
            self._bssf = met
            #model speichern            
            model.save('best-model.h5')

    def _handle_broken_model(self, model, error):
        '''
        Dient der Behandlung nicht compilierbarer oder decodierbarer modelle

        Parameters
        --------------
        model: keras.model
            Modellobjekt
        error: Object
            fehlerfall bzw. Python Exceptionobjekt
 
        Returns
        --------------
        loss: float
            Metriken die zurückgegeben werden
        acc: float
            Metriken die zurückgegeben werden
        '''
        # Löschen des MOdellobjektes
        del model

        n = self.genome_handler.n_classes
        #versuch der extraktion von metriken
        loss = log_loss(np.concatenate(([1], np.zeros(n - 1))), np.ones(n) / n)
        accuracy = 1 / n
        # grabadge collector
        gc.collect()

        # zurücksetzen des Backends
        if K.backend() == 'tensorflow':
            K.clear_session()
            tf.reset_default_graph()

        print('Ein Fehler ist aufgetreten und das Modell ist nicht trainierbar:')
        #Ausgabe des Exceptionobjektes
        print(error)
        print('prüfe ob die Ressorucen der Umgebung ausreichend sind!')
        
        return loss, accuracy
    
    
    def _evaluate_population(self, members, epochs, fitness, igen, ngen):
        '''
        Evaluation der Population unter Nutzung 
        des Klassenobjekts _Populationen
  
        Parameters
        --------------
        members: List
            liste der mitglieder der population
        epochs: int
            anzahl epochen
        fitness: float
            metrik ziel (loss, acc)
        igen: int
            index der wie vielten generation
        ngen: int
            anzahl der generationen
                     
        Returns
        --------------
        Object: class _Population
            _Populations Objekt (class _Polpulation) mit den Parametern
        '''
        fit = []
        # iteration durch das member objekt sowie der erstellung 
        for imem, mem in enumerate(members):
            self._print_evaluation(imem, len(members), igen, ngen)
            #übergabe an die _evaluate methode um das Model zu fitten
            res = self._evaluate(mem, epochs)
            v = res[self._metric_index]
            del res
            fit.append(v)

        fit = np.array(fit)
        self._print_result(fit, igen)
        return _Population(members, fit, fitness, obj=self._objective)

    def _print_evaluation(self, imod, nmod, igen, ngen):
        '''
        StdOut. Einfache print Methode
        
        Parameters
        --------------
        imod: int
            index welches Modell in der Population geprinted wird
        nmod: int
            ANzahl der Modelle in der Popualtion
        igen: int
            index der wie vielten generation
        ngen: int
            anzahl der generationen
        '''
        fstr = '\nmodel {0}/{1} - generation {2}/{3}:\n'
        print(fstr.format(imod + 1, nmod, igen + 1, ngen))

    def _generate_random_population(self, size):
        '''
        Aufruf aus dem genome_handler objekt die methode zum generieren neuer Mitglieder bzw. genomes für eine Population
        
        Parameters
        --------------
        size: int
            Anzahl generierender Genome einer Population (SearchSpace & SearchStrategy)
                
        Returns
        --------------
        List: List
            List der generierten Genome
        '''
        return [self.genome_handler.generate() for _ in range(size)]

    def _print_result(self, fitness, generation):
        '''
        StdOut
        Einfache printmehtode für das Ergebnis einer Population nach der Evaluation
               
        Parameters
        --------------
        fitness: Float
            Metrik
        geneneration: int
            aktuelle Generation
        
        Returns
        --------------
        members: List
            Liste der mutierten Mitglieder der nächsten Population
        '''
        result_str = ('Generation {3}:\t\tbest {4}: {0:0.4f}\t\taverage:'
                      '{1:0.4f}\t\tstd: {2:0.4f}')
        print(result_str.format(self._metric_objective(fitness),
                                np.mean(fitness),
                                np.std(fitness),
                                generation + 1, self._metric))

    def _crossover(self, genome1, genome2):
        '''
        Einfaches CrossOver durch zufällige auswahl einzelner Abschnitte des Genomes bzw. zweier Genomes
            
        Parameters
        --------------
        genome1: List
            Elterngenome 1
        genome2: List
            Elterngenome 2
        Returns: List
            Kindgenome

        Returns
        --------------
        members: List
            Liste der mutierten Mitglieder der nächsten Population
        '''
        # zufällige auswahl
        cross_ind = rand.randint(0, len(genome1))
        #erzeugen des neuen kindgenomes
        child = genome1[:cross_ind] + genome2[cross_ind:]
        return child

    def _mutate(self, genome, generation):
        '''
        Der eigentliche mutationsaufruf mindestens 3 Mutationen sind pflicht
        oder die ganzzahlige Teilung durch die Generationsanzahl.

        Parameters
        --------------
        genome: List
            genome welches mutiert werden soll
        generation: int
            Generationszahl bzw. welche Generation es aktuell ist.
        
        Returns
        --------------
        genome: List
            mutation durch aufruf der genome_handler methode mutate (neues genome)
        '''
        # anzahl der Mutationen generieren mindestens 3
        # TODO: eventuell Anpassung der mindestmutationen
        num_mutations = max(3, generation // 4) #Floor division zum abschneiden der nachkommastellen
        # Mutationsrückgabe bzw. eines genomes
        return self.genome_handler.mutate(genome, num_mutations)

class _Population(object):
    '''
    Populationsklasse.
    '''

    def __len__(self):
        return len(self.members)

    def __init__(self, members, fitnesses, score, obj='max'):
        self.members = members
        scores = fitnesses - fitnesses.min()
        if scores.max() > 0:
            scores /= scores.max()
        if obj == 'min':
            scores = 1 - scores
        if score:
            self.scores = score(scores)
        else:
            self.scores = scores
        self.s_fit = sum(self.scores)

    def get_best(self, n):
        '''
        Gibt den besten einer Population zurück
        
        Parameters:
        --------------
        n: int
            Anzahl

        Returns:
        --------------
        List: List
            beste Genom
        '''
        # holt sich alle Mitglieder aus einer Population
        combined = [(self.members[i], self.scores[i])
                    for i in range(len(self.members))]
        # sortiert diese
        sorted(combined, key=(lambda x: x[1]), reverse=True)
        # gibt N Mitglieder zurück
        return [x[0] for x in combined[:n]]

    def select(self):
        '''
        Selektion des besten Models
        Returns:
        --------------
        genome: List
            beste Modell der generation
        '''
        dart = rand.uniform(0, self.s_fit)
        sum_fits = 0
        for i in range(len(self.members)):
            sum_fits += self.scores[i]
            if sum_fits >= dart:
                return self.members[i]