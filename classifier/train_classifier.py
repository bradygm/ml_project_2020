from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Input, Flatten
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys


def split_validation_set(x_data, y_data, val_split=0.8):
    idx = np.arange(x_data.shape[0])
    np.random.shuffle(idx)
    train_samples = int(val_split * x_data.shape[0])
    train_idx = idx[:train_samples]
    val_idx = idx[train_samples:]

    x_train, x_val = x_data[train_idx, :, :, :], x_data[val_idx, :, :, :]
    y_train, y_val = y_data[train_idx, :], y_data[val_idx, :]
    return x_train, y_train, x_val, y_val


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

x_data = np.load(sys.argv[1]).astype(float)
y_data = keras.utils.to_categorical(np.load(sys.argv[2]).astype(int), 3)

x_train, y_train, x_val, y_val = split_validation_set(x_data, y_data)

x_test = np.load(sys.argv[3]).astype(float)
y_test = keras.utils.to_categorical(np.load(sys.argv[4]).astype(int), 3)

# Tensorboard
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Early stopping
earlystop_callback = keras.callbacks.EarlyStopping(
    monitor='val_categorical_accuracy', min_delta=0.0001,
    patience=2)

# Build LeNet model
inputs = keras.Input(shape=(28, 28, 3), name='cropped_images')
conv1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 3))(inputs)
mpool1 = MaxPool2D()(conv1)
conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(mpool1)
mpool2 = MaxPool2D()(conv2)
conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(mpool2)
mpool3 = MaxPool2D()(conv3)
# dropout1 = Dropout(0.25)(mpool2)
f1 = Flatten()(mpool3)
f2 = Dense(128, activation='relu')(f1)
# f3 = Dense(64, activation='relu')(f2)
# f4 = Dense(32, activation='relu')(f3)
outputs = Dense(3, activation='softmax')(f2)

model = keras.Model(inputs=inputs, outputs=outputs, name='lenet_model')
model.summary()

METRICS = [
    keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
]

model.compile(optimizer=keras.optimizers.SGD(lr=0.01),
              loss="categorical_crossentropy",
              metrics=METRICS)

training_history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=10, shuffle=True, callbacks=[tensorboard_callback, earlystop_callback])
model.evaluate(x_test, y_test, verbose=2)

print("Average train loss: ", np.average(training_history.history['loss']))

model.reset_metrics()
# predictions = model.predict(x_test)
model.save('models/lenet_all.h5')