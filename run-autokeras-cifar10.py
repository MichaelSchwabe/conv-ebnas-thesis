from tensorflow.keras.datasets import cifar10
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, CSVLogger, ModelCheckpoint

import autokeras as ak

# Prepare the dataset.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)  # (50000, 32, 32, 3)
print(y_train.shape)  # (50000,)
print(y_train[:3])  

# Initialize the ImageClassifier.
#clf = ak.ImageClassifier(max_trials=3)

clf = ak.ImageClassifier(
    num_classes=10,
    #multi_label=True,
    #loss=None,
    #metrics=None,
    project_name="autokeras-cifar10",
    max_trials=10,
    #directory=None,
    objective="val_accuracy",
    tuner='bayesian', #'greedy', 'bayesian', 'hyperband' or 'random'
    #seed=None,
    #max_model_size=None,
    overwrite=False
    )


'''
    num_classes=None,
    multi_label=False,
    loss=None,
    metrics=None,
    project_name="image_classifier",
    max_trials=100,
    directory=None,
    objective="val_loss",
    tuner=None,
    overwrite=False,
    seed=None,
    max_model_size=None,
    **kwargs

'''
i=0
callbacks = [
                # TODO: Gebe Innovationen eine Chance patience=1 .. vielleicht 2 oder 3
                # entscheidet das nach der ersten Epoche die keine Verbesserung mit sich bringt abgebrochen wird ...
                #EarlyStopping(monitor='val_loss', patience=1, verbose=1)
                #### 
                # NEW
                #EarlyStopping(monitor='val_loss', patience=2, verbose=1),
                TensorBoard(log_dir='./autokeras-cifar10-tfboardlogs', histogram_freq=1, profile_batch = 100000000),# write_graph=True, write_images=True),
                CSVLogger('autokeras-cifar10/log-maxtrail400.csv', separator=",", append=True)
                #ModelCheckpoint(filepath='autokeras-cifar10/models/model.{'+str(i = i + 1)+'}{epoch:02d}-{val_loss:.4f}.h5',
                #                              monitor='val_loss',
                #                              verbose=1,
                #                              save_best_only=True)
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

clf.fit(x_train, y_train, epochs=5, callbacks=callbacks, validation_split=0.2)
#, validation_data=None, **kwargs)

# Search for the best model.
#clf.fit(x_train, y_train, epochs=10)
# Evaluate on the testing data.
print("Accuracy: {accuracy}".format(accuracy=clf.evaluate(x_test, y_test)))

best = clf.export_model()
print(best.summary())
best.save("autokeras-cifar10-best.h5")