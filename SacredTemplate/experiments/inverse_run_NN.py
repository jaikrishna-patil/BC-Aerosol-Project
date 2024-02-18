import os
import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error




sys.path.append("../src")
from utils.experiment import Bunch, make_experiment, make_experiment_tempfile


if __name__ == '__main__':
    experiment = make_experiment()


    @experiment.config
    def config():
        params = dict(
            split='full_run',
            epochs=1000,
            patience=100,
            hidden_layers=8,
            batch_size=32,
            hidden_units=512,
            kernel_initializer='he_normal',
            activation='relu',
            loss='mean_squared_error',
            optimizer='adam'
            #range=(10, 15)
        )


    @experiment.automain
    def main(params, _run):

        params = Bunch(params)

        #Load dataset
        df = pd.read_excel('../../data/database_new.xlsx')
        X = df.iloc[:, [0, 7, 32, 33]]
        Y = df.iloc[:, [2]]


        # Normalizaing Min max
        pt = PowerTransformer(method='box-cox')
        X_train_transformed = pt.fit_transform(X + 0.00000000001)
        X_test_transformed = pt.transform(X + 0.00000000001)

        print(pd.DataFrame({'cols': X.columns, 'box_cox_lambdas': pt.lambdas_}))

        #Build NN model

        #model = build_model()#params.actuvation
        model = Sequential()
        model.add(Input(shape=(4,)))
        for j in range(0, params.hidden_layers):
            model.add(Dense(params.hidden_units, kernel_initializer=params.kernel_initializer, activation='relu'))

        model.add(Dense(1, kernel_initializer=params.kernel_initializer, activation='linear'))

        #Compile model
        model.compile(loss=params.loss, optimizer=params.optimizer,
                      metrics=['mean_absolute_error'])

        print(model.summary())


        #Running and logging model plus Early stopping

        filepath = f"inverse_run_{_run._id}/best_model.hdf5"
        with make_experiment_tempfile('best_model.hdf5', _run, mode='wb', suffix='.hdf5') as model_file:
            checkpoint = ModelCheckpoint(model_file.name, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
            # # patient early stopping
            es = EarlyStopping(monitor='val_loss', patience=100, verbose=1)

            callback_list = [checkpoint, es]
            history = model.fit(X_train_transformed, Y, epochs=params.epochs, batch_size=params.batch_size, validation_split=0.2, callbacks=callback_list)

            # choose the best Weights for prediction

            #Save the model

            #Save metrics loss and val_loss
            #print(history.history.keys())
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs = len(loss)
            for epoch in range(0, epochs):

                # Log scalar wil log a single number. The string is the metrics name
                _run.log_scalar('Training loss', loss[epoch])
                _run.log_scalar('Validation loss', val_loss[epoch])

            #Use best model to predict
            weights_file = f'inverse_run_{_run._id}/best_model.hdf5'  # choose the best checkpoint
            model.load_weights(model_file.name)  # load it
            model.compile(loss=params.loss, optimizer=params.optimizer, metrics=[params.loss])
        # Evaluate

        Y_pred = model.predict(X_test_transformed)

        #logging Y_test values
        Y_test = pd.DataFrame(data=Y, columns=["fraction_of_coating"])
        #Y_test.reset_index(inplace=True, drop=True)
        for i in Y_test['fraction_of_coating']:
            _run.log_scalar('Actual fraction_of_coating', i)
        # for i in Y_test['q_sca']:
        #     _run.log_scalar('Actual q_sca', i)
        # for i in Y_test['g']:
        #     _run.log_scalar('Actual g', i)
        #logging predicted values
        Y_pred = pd.DataFrame(data=Y_pred, columns=["fraction_of_coating"])
        #print(Y_pred)
        for i in Y_pred['fraction_of_coating']:
            _run.log_scalar('Predicted fraction_of_coating', i)
        # for i in Y_pred['q_sca']:
        #     _run.log_scalar('Predicted q_sca', i)
        # for i in Y_pred['g']:
        #     _run.log_scalar('Predicted g', i)
        # logging difference between the two
        Y_diff = Y_test - Y_pred
        for i in Y_diff['fraction_of_coating']:
            _run.log_scalar('Absolute error fraction_of_coating', i)
        # for i in Y_diff['q_sca']:
        #     _run.log_scalar('Absolute error q_sca', i)
        # for i in Y_diff['g']:
        #     _run.log_scalar('Absolute error g', i)

        error = mean_absolute_error(Y_test, Y_pred)



        #error=error*100
        print('Mean absolute error on test set [fraction_of_coating]:-  ', error)
        _run.info['error'] = error

