from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Input, Flatten
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

x_train = np.load(sys.argv[1]).astype(float) / 255.0
y_train = keras.utils.to_categorical(np.load(sys.argv[2]).astype(int), 2)

x_test = np.load(sys.argv[3]).astype(float) / 255.0
y_test = keras.utils.to_categorical(np.load(sys.argv[4]).astype(int), 2)

# Tensorboard
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Early stopping
earlystop_callback = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', min_delta=0.0001,
    patience=2)

# Build LeNet model
inputs = keras.Input(shape=(28, 28, 3), name='cropped_images')
conv1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 3))(inputs)
mpool1 = MaxPool2D()(conv1)
conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(mpool1)
mpool2 = MaxPool2D()(conv2)
dropout1 = Dropout(0.25)(mpool2)
f1 = Flatten()(dropout1)
f2 = Dense(128, activation='relu')(f1)
outputs = Dense(2, activation='softmax')(f2)

model = keras.Model(inputs=inputs, outputs=outputs, name='lenet_model')
model.summary()

METRICS = [
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.BinaryAccuracy(name='accuracy')
]

model.compile(optimizer=keras.optimizers.Adadelta(),
              loss="categorical_crossentropy",
              metrics=METRICS)

training_history = model.fit(x_train, y_train, validation_split=0.25, batch_size=64, epochs=15, shuffle=True, callbacks=[tensorboard_callback, earlystop_callback])
model.evaluate(x_test, y_test, verbose=2)

print("Average test loss: ", np.average(training_history.history['loss']))

model.reset_metrics()
predictions = model.predict(x_test)
model.save('models/lenet_bird.h5')