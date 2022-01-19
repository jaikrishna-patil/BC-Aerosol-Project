import numpy as np

from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sb
import matplotlib.pyplot as plt

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
from tensorflow import keras
from keras_tuner import RandomSearch
from tensorflow.keras import layers















#optimizer= tf.keras.optimizers.Adam(lr= 0.001)
# model.compile(optimizer='adam',
#              loss={
#                  'q_abs': 'mean_absolute_percentage_error',
#                  'q_sca': 'mean_absolute_percentage_error',
#                  'g': 'mean_absolute_percentage_error'
#              },
#              metrics={
#                  'q_abs': 'mean_absolute_percentage_error',
#                  'q_sca': 'mean_absolute_percentage_error',
#                  'g': 'mean_absolute_percentage_error'
#              })






#history= NN_model.fit(X_train, Y_train, epochs=7, batch_size=32, validation_split = 0.2)

# latest=tf.train.latest_checkpoint(checkpoint_dir)
weights_file = 'random_split_with_min_max/Weights-448--1.34756.hdf5'
model.load_weights(weights_file) # load it
model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])

#model.save('random_split_model.h5')

# evaluate the model
train_loss, train_acc = model.evaluate(X_train, Y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print('Train loss: %.3f, Validation loss: %.3f' % (train_loss, test_loss))
# plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend()
plt.show()




