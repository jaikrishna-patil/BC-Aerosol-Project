import os
import random
import sys


import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error
import pickle



sys.path.append("../src")
from utils.experiment import Bunch, make_experiment, make_experiment_tempfile


if __name__ == '__main__':
    experiment = make_experiment()


    @experiment.config
    def config():
        params = dict(
            epochs=1000,
            patience=100,
            hidden_layers=2,
            batch_size=32,
            hidden_units=512,
            kernel_initializer='he_normal',
            #n_hidden=8,
            #dense_units=[416, 288, 256,256, 192,448,288,128, 352,224],
            #kernel_initializer=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'],
            activation='relu',
            loss='mean_squared_error'
            #range=(10, 15)
        )


    @experiment.automain
    def main(params, _run):

        params = Bunch(params)

        #Load dataset
        df = pd.read_excel('database_new.xlsx')
        X = df.iloc[:, [0, 3, 7, 24, 25, 26, 27]]
        #X = df.iloc[:, [0, 24, 25, 26, 27]]
        Y = df.iloc[:, [1, 2]]



        # Standardizing data and targets
        scaling_x = MinMaxScaler()
        scaling_y = MinMaxScaler()
        X = scaling_x.fit_transform(X)
        Y = scaling_y.fit_transform(Y)
        inverse_scalerfile_x = 'inverse_scaler_x.sav'
        inverse_scalerfile_y = 'inverse_scaler_y.sav'
        pickle.dump(scaling_x, open(inverse_scalerfile_x, 'wb'))
        pickle.dump(scaling_y, open(inverse_scalerfile_y, 'wb'))
        #Build NN model

        #model = build_model()#params.actuvation
        model = Sequential()
        model.add(Input(shape=(7,)))
        for j in range(0, params.hidden_layers):
            model.add(Dense(params.hidden_units, kernel_initializer=params.kernel_initializer, activation='relu'))

        model.add(Dense(2, kernel_initializer='glorot_normal', activation='sigmoid'))
        #Compile model
        model.compile(loss=params.loss, optimizer='adam',
                      metrics=['mean_absolute_error'])

        print(model.summary())


        #Running and logging model plus Early stopping

        filepath = f"inverse_run_{_run._id}/inverse_best_model.hdf5"
        with make_experiment_tempfile('inverse_best_model.hdf5', _run, mode='wb', suffix='.hdf5') as model_file:
            #print(model_file.name)
            checkpoint = ModelCheckpoint(model_file.name, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')

            # # patient early stopping
            es = EarlyStopping(monitor='val_loss', patience=params.patience, verbose=1)

            #log_csv = CSVLogger('fractal_dimension_loss_logs.csv', separator=',', append=False)

            callback_list = [checkpoint, es]
            history = model.fit(X, Y, epochs=params.epochs, batch_size=params.batch_size, validation_split=0.2, callbacks=callback_list)

            # choose the best Weights for prediction

            #Save the model

            #Save metrics loss and val_loss
            #print(history.history.keys())
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs = len(loss)
            print(epochs)
            for epoch in range(0, epochs):

                # Log scalar wil log a single number. The string is the metrics name
                _run.log_scalar('Training loss', loss[epoch])
                _run.log_scalar('Validation loss', val_loss[epoch])

            #Use best model to predict
            weights_file = f'inverse_run_{_run._id}/inverse_best_model.hdf5'  # choose the best checkpoint
            model.load_weights(model_file.name)  # load it
            model.compile(loss=params.loss, optimizer='adam', metrics=[params.loss])
        #Evaluate plus inverse transforms
        Y_pred = model.predict(X)
        Y_test = scaling_y.inverse_transform(Y)
        Y_pred = scaling_y.inverse_transform(Y_pred)

        #logging Y_test values
        Y_test = pd.DataFrame(data=Y_test, columns=["fractal_dimension", "fraction_of_coating"])
        #Y_test.reset_index(inplace=True, drop=True)
        for i in Y_test['fractal_dimension']:
            _run.log_scalar('Actual fractal_dimension', i)
        for i in Y_test['fraction_of_coating']:
            _run.log_scalar('Actual fraction_of_coating', i)

        #logging predicted values
        Y_pred = pd.DataFrame(data=Y_pred, columns=["fractal_dimension", "fraction_of_coating"])
        for i in Y_pred['fractal_dimension']:
            _run.log_scalar('Predicted fractal_dimension', i)
        for i in Y_pred['fraction_of_coating']:
            _run.log_scalar('Predicted fraction_of_coating', i)

        # logging difference between the two
        Y_diff = Y_test - Y_pred
        for i in Y_diff['fractal_dimension']:
            _run.log_scalar('Absolute error fractal_dimension', i)
        for i in Y_diff['fraction_of_coating']:
            _run.log_scalar('Absolute error fraction_of_coating', i)

        error = mean_absolute_error(Y_test, Y_pred, multioutput='raw_values')
        _run.info['error'] = error



        #error=error*100
        print('Mean absolute error on test set [fractal_dimension, fraction_of_coating]:-  ', error)



