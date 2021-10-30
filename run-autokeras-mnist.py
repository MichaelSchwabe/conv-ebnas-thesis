from tensorflow.keras.datasets import mnist
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, CSVLogger

import autokeras as ak

# Prepare the dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(y_train[:3])  # array([7, 2, 1], dtype=uint8)

# Initialize the ImageClassifier.
#clf = ak.ImageClassifier(max_trials=3)

clf = ak.ImageClassifier(
    num_classes=10,
    #multi_label=True,
    #loss=None,
    #metrics=None,
    project_name="autokeras-mnist",
    max_trials=10,
    #directory=None,
    objective="val_accuracy",
    tuner='bayesian',
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

callbacks = [
                # TODO: Gebe Innovationen eine Chance patience=1 .. vielleicht 2 oder 3
                # entscheidet das nach der ersten Epoche die keine Verbesserung mit sich bringt abgebrochen wird ...
                #EarlyStopping(monitor='val_loss', patience=1, verbose=1)
                #### 
                # NEW
                #EarlyStopping(monitor='val_loss', patience=2, verbose=1),
                CSVLogger('autokeras-mnist/log-maxtrail400.csv', separator=",", append=True),
                TensorBoard(log_dir='./autokeras-mnist-tfboardlogs', histogram_freq=1, profile_batch = 100000000)#, write_graph=True, write_images=True)
                ####
                # TensorBoard(log_dir='./tfboardlogs')
                # histogram_freq: frequency (in epochs) at which to compute activation and weight histograms for the layers of the model. If set to 0, histograms won't be computed. Validation data (or split) must be specified for histogram visualizations.
                # write_graph: whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
                # write_images: whether to write model weights to visualize as image in TensorBoard.
                # 
                #
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
best.save("autokeras-mnist-best.h5")