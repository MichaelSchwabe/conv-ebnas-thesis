from tensorflow.keras.datasets import mnist
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, CSVLogger

import autokeras as ak

# Prepare the dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(y_train[:3])  # array([7, 2, 1], dtype=uint8)

# Initialize the ImageClassifier.
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


callbacks = [
                CSVLogger('autokeras-mnist/log-maxtrail400.csv', separator=",", append=True),
                TensorBoard(log_dir='./autokeras-mnist-tfboardlogs', histogram_freq=1, profile_batch = 100000000)#, write_graph=True, write_images=True)
            ]

clf.fit(x_train, y_train, epochs=5, callbacks=callbacks, validation_split=0.2)
#, validation_data=None, **kwargs)

print("Accuracy: {accuracy}".format(accuracy=clf.evaluate(x_test, y_test)))

best = clf.export_model()
print(best.summary())
best.save("autokeras-mnist-best.h5")