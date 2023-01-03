import os
import random
import sys

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from keras.regularizers import l2

sys.path.append("../src")
from utils.experiment import Bunch, make_experiment, make_experiment_tempfile



if __name__ == '__main__':
    experiment = make_experiment()


    @experiment.config
    def config():
        params = dict(
            split='fractal_dimension',
            split_type='interpolating',
            split_lower=-1,
            split_upper=-1,
            epochs=1000,
            patience=100,
            hidden_layers=2,
            batch_size=32,
            hidden_units=512,
            kernel_initializer='he_normal',
            # n_hidden=8,
            # dense_units=[416, 288, 256,256, 192,448,288,128, 352,224],
            # kernel_initializer=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'],
            activation='relu',
            loss='mean_squared_error'
            # range=(10, 15)
        )


    @experiment.automain
    def main(params, _run):

        params = Bunch(params)

        # Load dataset
        df = pd.read_excel('database_new.xlsx')
        X = df.iloc[:, :8]
        Y = df.iloc[:, 25:28]

        # Split on fractal dimension
        if params.split_type == 'interpolating':

            train_set = df[
                (df['fractal_dimension'] < params.split_lower) | (df['fractal_dimension'] > params.split_upper)]
            test_set = df[
                (df['fractal_dimension'] >= params.split_lower) & (df['fractal_dimension'] <= params.split_upper)]

        elif params.split_type == 'extrapolating_lower':
            train_set = df[(df['fractal_dimension'] > params.split_lower)]
            test_set = df[(df['fractal_dimension'] <= params.split_lower)]
        elif params.split_type == 'extrapolating_upper':
            train_set = df[(df['fractal_dimension'] < params.split_upper)]
            test_set = df[(df['fractal_dimension'] >= params.split_upper)]
        print(len(test_set))

        Y_train = train_set.iloc[:, 25:28]
        X_train = train_set.iloc[:, :8]
        Y_test = test_set.iloc[:, 25:28]
        X_test = test_set.iloc[:, :8]
        # Normalizaing Min max
        pt = PowerTransformer(method='box-cox')
        X_train_transformed = pt.fit_transform(X_train + 0.00000000001)
        X_test_transformed = pt.transform(X_test + 0.00000000001)

        print(pd.DataFrame({'cols': X_train.columns, 'box_cox_lambdas': pt.lambdas_}))

        #model = build_model()  # params.actuvation
        model = Sequential()
        model.add(Input(shape=(8,)))
        for j in range(0, params.hidden_layers):

            model.add(Dense(params.hidden_units, kernel_initializer=params.kernel_initializer, activation='relu'))


        model.add(Dense(3, kernel_initializer='glorot_normal', activation='sigmoid'))
        # Compile model
        model.compile(loss=params.loss, optimizer='adam',
                      metrics=['mean_absolute_error'])

        print(model.summary())

        # Running and logging model plus Early stopping

        filepath = f"fractal_dimension_{_run._id}/best_model.hdf5"
        with make_experiment_tempfile('best_model.hdf5', _run, mode='wb', suffix='.hdf5') as model_file:
            # print(model_file.name)
            checkpoint = ModelCheckpoint(model_file.name, verbose=1, monitor='val_loss', save_best_only=True,
                                         mode='auto')

            # # patient early stopping
            es = EarlyStopping(monitor='val_loss', patience=params.patience, verbose=1)

            # log_csv = CSVLogger('fractal_dimension_loss_logs.csv', separator=',', append=False)

            callback_list = [checkpoint, es]
            history = model.fit(X_train, Y_train, epochs=params.epochs, batch_size=params.batch_size,
                                validation_split=0.2, callbacks=callback_list)

            # choose the best Weights for prediction

            # Save the model

            # Save metrics loss and val_loss
            # print(history.history.keys())
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs = len(loss)
            print(epochs)
            for epoch in range(0, epochs):
                # Log scalar wil log a single number. The string is the metrics name
                _run.log_scalar('Training loss', loss[epoch])
                _run.log_scalar('Validation loss', val_loss[epoch])

            # Use best model to predict
            weights_file = f'fractal_dimension_{_run._id}/best_model.hdf5'  # choose the best checkpoint
            model.load_weights(model_file.name)  # load it
            model.compile(loss=params.loss, optimizer='adam', metrics=[params.loss])
        # Evaluate plus inverse transforms

        # Inverse transform
        Y_test = scaling_y.inverse_transform(Y_test)
        Y_pred = model.predict(X_test)
        Y_pred = scaling_y.inverse_transform(Y_pred)

        # logging Y_test values
        Y_test = pd.DataFrame(data=Y_test, columns=["q_abs", "q_sca", "g"])
        # Y_test.reset_index(inplace=True, drop=True)
        for i in Y_test['q_abs']:
            _run.log_scalar('Actual q_abs', i)
        for i in Y_test['q_sca']:
            _run.log_scalar('Actual q_sca', i)
        for i in Y_test['g']:
            _run.log_scalar('Actual g', i)
        # logging predicted values
        Y_pred = pd.DataFrame(data=Y_pred, columns=["q_abs", "q_sca", "g"])
        for i in Y_pred['q_abs']:
            _run.log_scalar('Predicted q_abs', i)
        for i in Y_pred['q_sca']:
            _run.log_scalar('Predicted q_sca', i)
        for i in Y_pred['g']:
            _run.log_scalar('Predicted g', i)
        # logging difference between the two
        Y_diff = Y_test - Y_pred
        for i in Y_diff['q_abs']:
            _run.log_scalar('Absolute error q_abs', i)
        for i in Y_diff['q_sca']:
            _run.log_scalar('Absolute error q_sca', i)
        for i in Y_diff['g']:
            _run.log_scalar('Absolute error g', i)

        error = mean_absolute_error(Y_test, Y_pred, multioutput='raw_values')

        # error=error*100
        print('Mean absolute error on test set [q_abs, q_sca, g]:-  ', error)
        _run.info['error'] = error
