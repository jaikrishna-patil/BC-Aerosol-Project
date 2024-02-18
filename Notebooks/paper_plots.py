import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np


from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, max_error, mean_absolute_percentage_error

import math

from sklearn.preprocessing import PowerTransformer
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_excel('database_new.xlsx')


def min_max_error_efficiency(y_test, y_pred):
    abs_error = abs(y_test - y_pred)
    max_error, efficiency_max = np.max(abs_error), y_test[np.argmax(abs_error)]
    min_error, efficiency_min = np.min(abs_error), y_test[np.argmin(abs_error)]

    return max_error, efficiency_max, min_error, efficiency_min

X = df.iloc[:, :8]
Y = df.iloc[:, 25:28]
X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y,
            test_size=0.30,
            random_state=10)
pt = PowerTransformer(method='box-cox')

X_test=X_test.reset_index(drop=True)
Y_test=Y_test.reset_index(drop=True)

X_train_transformed = pt.fit_transform(X_train+0.00000000001)
X_test_transformed = pt.transform(X_test+0.000000001)




print(pd.DataFrame({'cols':X_train.columns,'box_cox_lambdas':pt.lambdas_}))

regressor = KernelRidge(alpha=0.0001, gamma=0.75, kernel='rbf')


model = regressor.fit(X_train_transformed,Y_train)


Y_pred = model.predict(X_test_transformed)


Y_pred_KRR = pd.DataFrame(data=Y_pred, columns=["q_abs_pred_KRR", "q_sca_pred_KRR","g_pred_KRR"])
#Y_test.reset_index(inplace=True)
#Y_test.drop('index',axis=1, inplace=True)
mean_abs_error = mean_absolute_error(Y_test, Y_pred_KRR, multioutput='raw_values')

max_error_q_abs, q_abs_max, min_error_q_abs, q_abs_min = min_max_error_efficiency(Y_test['q_abs'], Y_pred_KRR['q_abs_pred_KRR'])
max_error_q_sca, q_sca_max, min_error_q_sca, q_sca_min = min_max_error_efficiency(Y_test['q_sca'], Y_pred_KRR['q_sca_pred_KRR'])
max_error_g, g_max, min_error_g, g_min = min_max_error_efficiency(Y_test['g'], Y_pred_KRR['g_pred_KRR'])

mape  = mean_absolute_percentage_error(Y_test, Y_pred_KRR, multioutput='raw_values')

# error=calculate_mean_absolute_percentage_error_multi(parameter_alpha, parameter_kernel# parameter_gamma, X_train, Y_train, X_test, Y_test, scaling_y)
print('Mean absolute error on test set: ', mean_abs_error)
print(f'Max error is {max_error_q_abs} for  q_abs {q_abs_max} on test set')
print(f'Min error is {min_error_q_abs} for  q_abs {q_abs_min} on test set')

print(f'Max error is {max_error_q_sca} for  q_sca {q_sca_max} on test set')
print(f'Min error is {min_error_q_sca} for  q_sca {q_sca_min} on test set')

print(f'Max error is {max_error_g} for  g {g_max} on test set')
print(f'Min error is {min_error_g} for  g {g_min} on test set')


print('Mean absolute percentage error on test set: ', mape)
# Running and logging model plus Early stopping

# df_new= df.sort_values(by = ['equi_mobility_dia'])

X = df.iloc[:, :8]
Y = df.iloc[:, 25:28]
X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y,
            test_size=0.30,
            random_state=10)
pt = PowerTransformer(method='box-cox')

X_test=X_test.reset_index(drop=True)
Y_test=Y_test.reset_index(drop=True)

X_train_transformed = pt.fit_transform(X_train+0.00000000001)
X_test_transformed = pt.transform(X_test+0.000000001)


print(pd.DataFrame({'cols':X_train.columns,'box_cox_lambdas':pt.lambdas_}))


model = load_model('best_model_forward_random.hdf5')
Y_pred_NN = model.predict(X_test_transformed)

Y_test = pd.DataFrame(data=Y_test, columns=["q_abs", "q_sca","g"])

Y_pred_NN = pd.DataFrame(data=Y_pred_NN, columns=["q_abs_pred_NN", "q_sca_pred_NN","g_pred_NN"])
#Y_test.reset_index(inplace=True)
#Y_test.drop('index',axis=1, inplace=True)
mean_abs_error = mean_absolute_error(Y_test, Y_pred_NN, multioutput='raw_values')
max_error_q_abs, q_abs_max, min_error_q_abs, q_abs_min = min_max_error_efficiency(Y_test['q_abs'], Y_pred_NN['q_abs_pred_NN'])
max_error_q_sca, q_sca_max, min_error_q_sca, q_sca_min = min_max_error_efficiency(Y_test['q_sca'], Y_pred_NN['q_sca_pred_NN'])
max_error_g, g_max, min_error_g, g_min = min_max_error_efficiency(Y_test['g'], Y_pred_NN['g_pred_NN'])

mape  = mean_absolute_percentage_error(Y_test, Y_pred_KRR, multioutput='raw_values')

# error=calculate_mean_absolute_percentage_error_multi(parameter_alpha, parameter_kernel# parameter_gamma, X_train, Y_train, X_test, Y_test, scaling_y)
print('Mean absolute error on test set: ', mean_abs_error)
print(f'Max error is {max_error_q_abs} for  q_abs {q_abs_max} on test set')
print(f'Min error is {min_error_q_abs} for  q_abs {q_abs_min} on test set')

print(f'Max error is {max_error_q_sca} for  q_sca {q_sca_max} on test set')
print(f'Min error is {min_error_q_sca} for  q_sca {q_sca_min} on test set')

print(f'Max error is {max_error_g} for  g {g_max} on test set')
print(f'Min error is {min_error_g} for  g {g_min} on test set')


print('Mean absolute percentage error on test set: ', mape)
df_random=pd.concat([X_test, Y_test, Y_pred_KRR, Y_pred_NN], axis =1)
df_random= df_random.sort_values(by=['equi_mobility_dia'])
df_random=df_random.reset_index(drop=True)
print(df_random)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
text_width = 6.96

def setup_matplotlib(small_size: int = SMALL_SIZE, medium_size: int = MEDIUM_SIZE):
    # Use LaTeX to typeset all text in the figure
    # This obviously needs a working LaTeX installation on the system
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': small_size,
        'axes.titlesize': medium_size,
        'axes.labelsize': medium_size,
        'xtick.labelsize': medium_size,
        'ytick.labelsize': medium_size,
        'legend.fontsize': medium_size,
        'figure.titlesize': medium_size,
        'text.usetex': True,
        'mathtext.fontset': 'cm',
        'mathtext.rm': 'serif',
        'text.latex.preamble': '\\usepackage{amsmath}\n'
                               '\\usepackage{amssymb}'
})

setup_matplotlib()

df = pd.read_excel('database_new.xlsx')

train_set = df[(df['fractal_dimension'] < 2.5)]
test_set = df[(df['fractal_dimension'] >= 2.5)]
Y_train = train_set.iloc[:, 25:28]
X_train = train_set.iloc[:, :8]
Y_test = test_set.iloc[:, 25:28]
X_test = test_set.iloc[:, :8]

pt = PowerTransformer(method='box-cox')

X_test=X_test.reset_index(drop=True)
Y_test=Y_test.reset_index(drop=True)

X_train_transformed = pt.fit_transform(X_train+0.00000000001)
X_test_transformed = pt.transform(X_test+0.000000001)




print(pd.DataFrame({'cols':X_train.columns,'box_cox_lambdas':pt.lambdas_}))

regressor = KernelRidge(alpha=0.001, gamma=0.01, kernel='rbf')


model = regressor.fit(X_train_transformed,Y_train)


Y_pred = model.predict(X_test_transformed)


Y_pred_KRR = pd.DataFrame(data=Y_pred, columns=["q_abs_pred_KRR", "q_sca_pred_KRR","g_pred_KRR"])
#Y_test.reset_index(inplace=True)
#Y_test.drop('index',axis=1, inplace=True)
error = mean_absolute_error(Y_test, Y_pred_KRR, multioutput='raw_values')
max_error_q_abs, q_abs_max, min_error_q_abs, q_abs_min = min_max_error_efficiency(Y_test['q_abs'], Y_pred_KRR['q_abs_pred_KRR'])
max_error_q_sca, q_sca_max, min_error_q_sca, q_sca_min = min_max_error_efficiency(Y_test['q_sca'], Y_pred_KRR['q_sca_pred_KRR'])
max_error_g, g_max, min_error_g, g_min = min_max_error_efficiency(Y_test['g'], Y_pred_KRR['g_pred_KRR'])

mape  = mean_absolute_percentage_error(Y_test, Y_pred_KRR, multioutput='raw_values')

# error=calculate_mean_absolute_percentage_error_multi(parameter_alpha, parameter_kernel# parameter_gamma, X_train, Y_train, X_test, Y_test, scaling_y)
print('Mean absolute error on test set: ', mean_abs_error)
print(f'Max error is {max_error_q_abs} for  q_abs {q_abs_max} on test set')
print(f'Min error is {min_error_q_abs} for  q_abs {q_abs_min} on test set')

print(f'Max error is {max_error_q_sca} for  q_sca {q_sca_max} on test set')
print(f'Min error is {min_error_q_sca} for  q_sca {q_sca_min} on test set')

print(f'Max error is {max_error_g} for  g {g_max} on test set')
print(f'Min error is {min_error_g} for  g {g_min} on test set')


print('Mean absolute percentage error on test set: ', mape)
# Running and logging model plus Early stopping

#Neural network for extrapolation
from keras.models import Sequential
from keras.layers import Dense, Input, LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger


df = pd.read_excel('database_new.xlsx')
train_set = df[(df['fractal_dimension'] < 2.5)]
test_set = df[(df['fractal_dimension'] >= 2.5)]
Y_train = train_set.iloc[:, 25:28]
X_train = train_set.iloc[:, :8]
Y_test = test_set.iloc[:, 25:28]
X_test = test_set.iloc[:, :8]
X_test=X_test.reset_index(drop=True)
Y_test=Y_test.reset_index(drop=True)
pt = PowerTransformer(method='box-cox')

X_test=X_test.reset_index(drop=True)
Y_test=Y_test.reset_index(drop=True)

X_train_transformed = pt.fit_transform(X_train+0.00000000001)
X_test_transformed = pt.transform(X_test+0.000000001)


print(pd.DataFrame({'cols':X_train.columns,'box_cox_lambdas':pt.lambdas_}))


model = load_model('extrapolate_fwd_fractal_dim/best_model.hdf5')
Y_pred_NN = model.predict(X_test_transformed)

Y_test = pd.DataFrame(data=Y_test, columns=["q_abs", "q_sca","g"])

Y_pred_NN = pd.DataFrame(data=Y_pred_NN, columns=["q_abs_pred_NN", "q_sca_pred_NN","g_pred_NN"])
#Y_test.reset_index(inplace=True)
#Y_test.drop('index',axis=1, inplace=True)
error = mean_absolute_error(Y_test, Y_pred_NN, multioutput='raw_values')
max_error_q_abs, q_abs_max, min_error_q_abs, q_abs_min = min_max_error_efficiency(Y_test['q_abs'], Y_pred_NN['q_abs_pred_NN'])
max_error_q_sca, q_sca_max, min_error_q_sca, q_sca_min = min_max_error_efficiency(Y_test['q_sca'], Y_pred_NN['q_sca_pred_NN'])
max_error_g, g_max, min_error_g, g_min = min_max_error_efficiency(Y_test['g'], Y_pred_NN['g_pred_NN'])

mape  = mean_absolute_percentage_error(Y_test, Y_pred_KRR, multioutput='raw_values')

# error=calculate_mean_absolute_percentage_error_multi(parameter_alpha, parameter_kernel# parameter_gamma, X_train, Y_train, X_test, Y_test, scaling_y)
print('Mean absolute error on test set: ', mean_abs_error)
print(f'Max error is {max_error_q_abs} for  q_abs {q_abs_max} on test set')
print(f'Min error is {min_error_q_abs} for  q_abs {q_abs_min} on test set')

print(f'Max error is {max_error_q_sca} for  q_sca {q_sca_max} on test set')
print(f'Min error is {min_error_q_sca} for  q_sca {q_sca_min} on test set')

print(f'Max error is {max_error_g} for  g {g_max} on test set')
print(f'Min error is {min_error_g} for  g {g_min} on test set')


print('Mean absolute percentage error on test set: ', mape)


df_extrapol=pd.concat([X_test, Y_test, Y_pred_KRR, Y_pred_NN], axis =1)
df_extrapol= df_extrapol.sort_values(by=['equi_mobility_dia'])
df_extrapol=df_extrapol.reset_index(drop=True)
print(df_extrapol)


train_set = df[(df['fractal_dimension'] < 2.1)|(df['fractal_dimension'] > 2.5)]
test_set = df[(df['fractal_dimension'] >= 2.1)&(df['fractal_dimension'] <= 2.5)]
Y_train = train_set.iloc[:, 25:28]
X_train = train_set.iloc[:, :8]
Y_test = test_set.iloc[:, 25:28]
X_test = test_set.iloc[:, :8]

pt = PowerTransformer(method='box-cox')

X_test=X_test.reset_index(drop=True)
Y_test=Y_test.reset_index(drop=True)

X_train_transformed = pt.fit_transform(X_train+0.00000000001)
X_test_transformed = pt.transform(X_test+0.000000001)




print(pd.DataFrame({'cols':X_train.columns,'box_cox_lambdas':pt.lambdas_}))

regressor = KernelRidge(alpha=0.001, gamma=0.01, kernel='rbf')


model = regressor.fit(X_train_transformed,Y_train)


Y_pred = model.predict(X_test_transformed)


Y_pred_KRR = pd.DataFrame(data=Y_pred, columns=["q_abs_pred_KRR", "q_sca_pred_KRR","g_pred_KRR"])
#Y_test.reset_index(inplace=True)
#Y_test.drop('index',axis=1, inplace=True)
error = mean_absolute_error(Y_test, Y_pred_KRR, multioutput='raw_values')
max_error_q_abs, q_abs_max, min_error_q_abs, q_abs_min = min_max_error_efficiency(Y_test['q_abs'], Y_pred_KRR['q_abs_pred_KRR'])
max_error_q_sca, q_sca_max, min_error_q_sca, q_sca_min = min_max_error_efficiency(Y_test['q_sca'], Y_pred_KRR['q_sca_pred_KRR'])
max_error_g, g_max, min_error_g, g_min = min_max_error_efficiency(Y_test['g'], Y_pred_KRR['g_pred_KRR'])

mape  = mean_absolute_percentage_error(Y_test, Y_pred_KRR, multioutput='raw_values')

# error=calculate_mean_absolute_percentage_error_multi(parameter_alpha, parameter_kernel# parameter_gamma, X_train, Y_train, X_test, Y_test, scaling_y)
print('Mean absolute error on test set: ', mean_abs_error)
print(f'Max error is {max_error_q_abs} for  q_abs {q_abs_max} on test set')
print(f'Min error is {min_error_q_abs} for  q_abs {q_abs_min} on test set')

print(f'Max error is {max_error_q_sca} for  q_sca {q_sca_max} on test set')
print(f'Min error is {min_error_q_sca} for  q_sca {q_sca_min} on test set')

print(f'Max error is {max_error_g} for  g {g_max} on test set')
print(f'Min error is {min_error_g} for  g {g_min} on test set')


print('Mean absolute percentage error on test set: ', mape)
# Running and logging model plus Early stopping
#Neural network for extrapolation
from keras.models import Sequential
from keras.layers import Dense, Input, LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

df = pd.read_excel('database_new.xlsx')
train_set = df[(df['fractal_dimension'] < 2.1)|(df['fractal_dimension'] > 2.5)]
test_set = df[(df['fractal_dimension'] >= 2.1)&(df['fractal_dimension'] <= 2.5)]
Y_train = train_set.iloc[:, 25:28]
X_train = train_set.iloc[:, :8]
Y_test = test_set.iloc[:, 25:28]
X_test = test_set.iloc[:, :8]
X_test=X_test.reset_index(drop=True)
Y_test=Y_test.reset_index(drop=True)
pt = PowerTransformer(method='box-cox')

X_test=X_test.reset_index(drop=True)
Y_test=Y_test.reset_index(drop=True)

X_train_transformed = pt.fit_transform(X_train+0.00000000001)
X_test_transformed = pt.transform(X_test+0.000000001)


print(pd.DataFrame({'cols':X_train.columns,'box_cox_lambdas':pt.lambdas_}))


model = load_model('interpolate_fwd_fractal_dim/best_model.hdf5')
Y_pred_NN = model.predict(X_test_transformed)

Y_test = pd.DataFrame(data=Y_test, columns=["q_abs", "q_sca","g"])

Y_pred_NN = pd.DataFrame(data=Y_pred_NN, columns=["q_abs_pred_NN", "q_sca_pred_NN","g_pred_NN"])
#Y_test.reset_index(inplace=True)
#Y_test.drop('index',axis=1, inplace=True)
error = mean_absolute_error(Y_test, Y_pred_NN, multioutput='raw_values')
max_error_q_abs, q_abs_max, min_error_q_abs, q_abs_min = min_max_error_efficiency(Y_test['q_abs'], Y_pred_NN['q_abs_pred_NN'])
max_error_q_sca, q_sca_max, min_error_q_sca, q_sca_min = min_max_error_efficiency(Y_test['q_sca'], Y_pred_NN['q_sca_pred_NN'])
max_error_g, g_max, min_error_g, g_min = min_max_error_efficiency(Y_test['g'], Y_pred_NN['g_pred_NN'])

mape  = mean_absolute_percentage_error(Y_test, Y_pred_KRR, multioutput='raw_values')

# error=calculate_mean_absolute_percentage_error_multi(parameter_alpha, parameter_kernel# parameter_gamma, X_train, Y_train, X_test, Y_test, scaling_y)
print('Mean absolute error on test set: ', mean_abs_error)
print(f'Max error is {max_error_q_abs} for  q_abs {q_abs_max} on test set')
print(f'Min error is {min_error_q_abs} for  q_abs {q_abs_min} on test set')

print(f'Max error is {max_error_q_sca} for  q_sca {q_sca_max} on test set')
print(f'Min error is {min_error_q_sca} for  q_sca {q_sca_min} on test set')

print(f'Max error is {max_error_g} for  g {g_max} on test set')
print(f'Min error is {min_error_g} for  g {g_min} on test set')


print('Mean absolute percentage error on test set: ', mape)

df_interpol=pd.concat([X_test, Y_test, Y_pred_KRR, Y_pred_NN], axis =1)
df_interpol= df_interpol.sort_values(by=['equi_mobility_dia'])
df_interpol=df_interpol.reset_index(drop=True)
print(df_interpol)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def create_boxplots(df_random, df_extrapol, df_interpol, feature_x):
    fig, axs = plt.subplots(3, 3, figsize=(1 * text_width, 0.9 * text_width), sharex='col', sharey='row')

    data_KRR = []
    data_NN = []
    ticks = []
    lower_val = 1.5
    upper_val = 1.7
    mstm_values = df_random[(df_random[feature_x] >= lower_val) & (df_random[feature_x] <= upper_val)]['q_abs']
    KRR_values = df_random[(df_random[feature_x] >= lower_val) & (df_random[feature_x] <= upper_val)]['q_abs_pred_KRR']
    NN_values = df_random[(df_random[feature_x] >= lower_val) & (df_random[feature_x] <= upper_val)]['q_abs_pred_NN']
    l1 = KRR_values.to_numpy() - mstm_values
    l2 = NN_values.to_numpy() - mstm_values
    data_KRR.append(l1)
    data_NN.append(l2)
    ticks.append(f'{lower_val}-{upper_val}')
    lower_val = 1.7
    upper_val = 2.3
    for i in range(0, 2):
        mstm_values = df_random[(df_random[feature_x] > lower_val) & (df_random[feature_x] <= upper_val)]['q_abs']
        KRR_values = df_random[(df_random[feature_x] > lower_val) & (df_random[feature_x] <= upper_val)][
            'q_abs_pred_KRR']
        NN_values = df_random[(df_random[feature_x] > lower_val) & (df_random[feature_x] <= upper_val)]['q_abs_pred_NN']
        l1 = KRR_values.to_numpy() - mstm_values
        l2 = NN_values.to_numpy() - mstm_values
        data_KRR.append(l1)
        data_NN.append(l2)
        ticks.append(f'{lower_val + 0.2}-{upper_val}')
        lower_val = 2.3
        upper_val = 2.9

    # Creating plot
    bpl = axs[0, 0].boxplot(data_KRR, labels=ticks, positions=np.array(range(len(data_KRR))) * 2.0 - 0.4, sym="")
    bpr = axs[0, 0].boxplot(data_NN, labels=ticks, positions=np.array(range(len(data_NN))) * 2.0 + 0.4, sym="")
    axs[0, 0].grid(linestyle="dashed")
    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')
    axs[0, 0].plot([], c='#D7191C')
    axs[0, 0].plot([], c='#2C7BB6')
    axs[0, 0].set_xticks(range(0, len(ticks) * 2, 2), ticks)
    axs[0, 0].set(ylabel='$\hat{Q}_{abs}-Q_{abs}$')
    axs[0, 0].title.set_text('Random split')

    ########
    data_KRR = []
    data_NN = []
    ticks = []
    lower_val = 2.1
    #     upper_val=1.9
    mstm_values = df_interpol[(df_interpol[feature_x] == lower_val)]['q_abs']
    KRR_values = df_interpol[(df_interpol[feature_x] == lower_val)]['q_abs_pred_KRR']
    NN_values = df_interpol[(df_interpol[feature_x] == lower_val)]['q_abs_pred_NN']
    l1 = KRR_values.to_numpy() - mstm_values
    l2 = NN_values.to_numpy() - mstm_values
    data_KRR.append(l1)
    data_NN.append(l2)
    ticks.append(f'{lower_val}')
    lower_val = 2.3
    #     upper_val=2.5
    for i in range(0, 2):
        mstm_values = df_interpol[(df_interpol[feature_x] == lower_val)]['q_abs']
        KRR_values = df_interpol[(df_interpol[feature_x] == lower_val)]['q_abs_pred_KRR']
        NN_values = df_interpol[(df_interpol[feature_x] == lower_val)]['q_abs_pred_NN']
        l1 = KRR_values.to_numpy() - mstm_values
        l2 = NN_values.to_numpy() - mstm_values
        data_KRR.append(l1)
        data_NN.append(l2)
        ticks.append(f'{lower_val}')
        lower_val = 2.5

    # Creating plot
    bpl = axs[0, 1].boxplot(data_KRR, labels=ticks, positions=np.array(range(len(data_KRR))) * 2.0 - 0.4, sym="")
    bpr = axs[0, 1].boxplot(data_NN, labels=ticks, positions=np.array(range(len(data_NN))) * 2.0 + 0.4, sym="")
    axs[0, 1].grid(linestyle="dashed")
    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')
    axs[0, 1].plot([], c='#D7191C', label='KRR')
    axs[0, 1].plot([], c='#2C7BB6', label='NN')
    axs[0, 1].set_xticks(range(0, len(ticks) * 2, 2), ticks)
    axs[0, 1].title.set_text('Interpolation split')
    #     axs[0,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ######
    data_KRR = []
    data_NN = []
    ticks = []
    lower_val = 2.7
    #     upper_val=1.7
    mstm_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['q_abs']
    KRR_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['q_abs_pred_KRR']
    NN_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['q_abs_pred_NN']
    l1 = KRR_values.to_numpy() - mstm_values
    l2 = NN_values.to_numpy() - mstm_values
    data_KRR.append(l1)
    data_NN.append(l2)
    ticks.append(f'{lower_val}')
    #     lower_val=2.9
    lower_val = 2.9
    for i in range(0, 1):
        mstm_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['q_abs']
        KRR_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['q_abs_pred_KRR']
        NN_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['q_abs_pred_NN']
        l1 = KRR_values.to_numpy() - mstm_values
        l2 = NN_values.to_numpy() - mstm_values
        data_KRR.append(l1)
        data_NN.append(l2)
        ticks.append(f'{lower_val}')
    #         lower_val=2.3
    #         upper_val=2.9

    # Creating plot
    bpl = axs[0, 2].boxplot(data_KRR, labels=ticks, positions=np.array(range(len(data_KRR))) * 2.0 - 0.4, sym="")
    bpr = axs[0, 2].boxplot(data_NN, labels=ticks, positions=np.array(range(len(data_NN))) * 2.0 + 0.4, sym="")
    axs[0, 2].grid(linestyle="dashed")
    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')
    axs[0, 2].plot([], c='#D7191C')
    axs[0, 2].plot([], c='#2C7BB6')
    axs[0, 2].set_xticks(range(0, len(ticks) * 2, 2), ticks)
    axs[0, 2].title.set_text('Extrapolation split')
    ##################################################################
    data_KRR = []
    data_NN = []
    ticks = []
    lower_val = 1.5
    upper_val = 1.7
    mstm_values = df_random[(df_random[feature_x] >= lower_val) & (df_random[feature_x] <= upper_val)]['q_sca']
    KRR_values = df_random[(df_random[feature_x] >= lower_val) & (df_random[feature_x] <= upper_val)]['q_sca_pred_KRR']
    NN_values = df_random[(df_random[feature_x] >= lower_val) & (df_random[feature_x] <= upper_val)]['q_sca_pred_NN']
    l1 = KRR_values.to_numpy() - mstm_values
    l2 = NN_values.to_numpy() - mstm_values
    data_KRR.append(l1)
    data_NN.append(l2)
    ticks.append(f'{lower_val}-{upper_val}')
    lower_val = 1.7
    upper_val = 2.3
    for i in range(0, 2):
        mstm_values = df_random[(df_random[feature_x] > lower_val) & (df_random[feature_x] <= upper_val)]['q_sca']
        KRR_values = df_random[(df_random[feature_x] > lower_val) & (df_random[feature_x] <= upper_val)][
            'q_sca_pred_KRR']
        NN_values = df_random[(df_random[feature_x] > lower_val) & (df_random[feature_x] <= upper_val)]['q_sca_pred_NN']
        l1 = KRR_values.to_numpy() - mstm_values
        l2 = NN_values.to_numpy() - mstm_values
        data_KRR.append(l1)
        data_NN.append(l2)
        ticks.append(f'{lower_val + 0.2}-{upper_val}')
        lower_val = 2.3
        upper_val = 2.9

    # Creating plot
    bpl = axs[1, 0].boxplot(data_KRR, labels=ticks, positions=np.array(range(len(data_KRR))) * 2.0 - 0.4, sym="")
    bpr = axs[1, 0].boxplot(data_NN, labels=ticks, positions=np.array(range(len(data_NN))) * 2.0 + 0.4, sym="")
    axs[1, 0].grid(linestyle="dashed")
    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')
    axs[1, 0].plot([], c='#D7191C')
    axs[1, 0].plot([], c='#2C7BB6')
    axs[1, 0].set_xticks(range(0, len(ticks) * 2, 2), ticks)
    axs[1, 0].set(ylabel='$\hat{Q}_{sca}-Q_{sca}$')

    ########
    data_KRR = []
    data_NN = []
    ticks = []
    lower_val = 2.1
    #     upper_val=1.9
    mstm_values = df_interpol[(df_interpol[feature_x] == lower_val)]['q_sca']
    KRR_values = df_interpol[(df_interpol[feature_x] == lower_val)]['q_sca_pred_KRR']
    NN_values = df_interpol[(df_interpol[feature_x] == lower_val)]['q_sca_pred_NN']
    l1 = KRR_values.to_numpy() - mstm_values
    l2 = NN_values.to_numpy() - mstm_values
    data_KRR.append(l1)
    data_NN.append(l2)
    ticks.append(f'{lower_val}')
    lower_val = 2.3
    #     upper_val=2.5
    for i in range(0, 2):
        mstm_values = df_interpol[(df_interpol[feature_x] == lower_val)]['q_sca']
        KRR_values = df_interpol[(df_interpol[feature_x] == lower_val)]['q_sca_pred_KRR']
        NN_values = df_interpol[(df_interpol[feature_x] == lower_val)]['q_sca_pred_NN']
        l1 = KRR_values.to_numpy() - mstm_values
        l2 = NN_values.to_numpy() - mstm_values
        data_KRR.append(l1)
        data_NN.append(l2)
        ticks.append(f'{lower_val}')
        lower_val = 2.5

    # Creating plot
    bpl = axs[1, 1].boxplot(data_KRR, labels=ticks, positions=np.array(range(len(data_KRR))) * 2.0 - 0.4, sym="")
    bpr = axs[1, 1].boxplot(data_NN, labels=ticks, positions=np.array(range(len(data_NN))) * 2.0 + 0.4, sym="")
    axs[1, 1].grid(linestyle="dashed")
    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')
    axs[1, 1].plot([], c='#D7191C')
    axs[1, 1].plot([], c='#2C7BB6')
    axs[1, 1].set_xticks(range(0, len(ticks) * 2, 2), ticks)

    ######
    data_KRR = []
    data_NN = []
    ticks = []
    lower_val = 2.7
    #     upper_val=1.7
    mstm_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['q_sca']
    KRR_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['q_sca_pred_KRR']
    NN_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['q_sca_pred_NN']
    l1 = KRR_values.to_numpy() - mstm_values
    l2 = NN_values.to_numpy() - mstm_values
    data_KRR.append(l1)
    data_NN.append(l2)
    ticks.append(f'{lower_val}')
    #     lower_val=2.9
    lower_val = 2.9
    for i in range(0, 1):
        mstm_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['q_sca']
        KRR_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['q_sca_pred_KRR']
        NN_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['q_sca_pred_NN']
        l1 = KRR_values.to_numpy() - mstm_values
        l2 = NN_values.to_numpy() - mstm_values
        data_KRR.append(l1)
        data_NN.append(l2)
        ticks.append(f'{lower_val}')
    #         lower_val=2.3
    #         upper_val=2.9

    # Creating plot
    bpl = axs[1, 2].boxplot(data_KRR, labels=ticks, positions=np.array(range(len(data_KRR))) * 2.0 - 0.4, sym="")
    bpr = axs[1, 2].boxplot(data_NN, labels=ticks, positions=np.array(range(len(data_NN))) * 2.0 + 0.4, sym="")
    axs[1, 2].grid(linestyle="dashed")
    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')
    axs[1, 2].plot([], c='#D7191C')
    axs[1, 2].plot([], c='#2C7BB6')
    axs[1, 2].set_xticks(range(0, len(ticks) * 2, 2), ticks)
    ################################################################################3
    data_KRR = []
    data_NN = []
    ticks = []
    lower_val = 1.5
    upper_val = 1.7
    mstm_values = df_random[(df_random[feature_x] >= lower_val) & (df_random[feature_x] <= upper_val)]['g']
    KRR_values = df_random[(df_random[feature_x] >= lower_val) & (df_random[feature_x] <= upper_val)]['g_pred_KRR']
    NN_values = df_random[(df_random[feature_x] >= lower_val) & (df_random[feature_x] <= upper_val)]['g_pred_NN']
    l1 = KRR_values.to_numpy() - mstm_values
    l2 = NN_values.to_numpy() - mstm_values
    data_KRR.append(l1)
    data_NN.append(l2)
    ticks.append(f'{lower_val}-{upper_val}')
    lower_val = 1.7
    upper_val = 2.3
    for i in range(0, 2):
        mstm_values = df_random[(df_random[feature_x] > lower_val) & (df_random[feature_x] <= upper_val)]['g']
        KRR_values = df_random[(df_random[feature_x] > lower_val) & (df_random[feature_x] <= upper_val)]['g_pred_KRR']
        NN_values = df_random[(df_random[feature_x] > lower_val) & (df_random[feature_x] <= upper_val)]['g_pred_NN']
        l1 = KRR_values.to_numpy() - mstm_values
        l2 = NN_values.to_numpy() - mstm_values
        data_KRR.append(l1)
        data_NN.append(l2)
        ticks.append(f'{lower_val + 0.2}-{upper_val}')
        lower_val = 2.3
        upper_val = 2.9

    # Creating plot
    bpl = axs[2, 0].boxplot(data_KRR, labels=ticks, positions=np.array(range(len(data_KRR))) * 2.0 - 0.4, sym="")
    bpr = axs[2, 0].boxplot(data_NN, labels=ticks, positions=np.array(range(len(data_NN))) * 2.0 + 0.4, sym="")
    axs[2, 0].grid(linestyle="dashed")
    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')
    axs[2, 0].plot([], c='#D7191C')
    axs[2, 0].plot([], c='#2C7BB6')
    axs[2, 0].set_xticks(range(0, len(ticks) * 2, 2), ticks)
    axs[2, 0].set(ylabel='$\hat{g}-g$')
    axs[2, 0].set(xlabel='$D_{f}$')

    ########
    data_KRR = []
    data_NN = []
    ticks = []
    lower_val = 2.1
    #     upper_val=1.9
    mstm_values = df_interpol[(df_interpol[feature_x] == lower_val)]['g']
    KRR_values = df_interpol[(df_interpol[feature_x] == lower_val)]['g_pred_KRR']
    NN_values = df_interpol[(df_interpol[feature_x] == lower_val)]['g_pred_NN']
    l1 = KRR_values.to_numpy() - mstm_values
    l2 = NN_values.to_numpy() - mstm_values
    data_KRR.append(l1)
    data_NN.append(l2)
    ticks.append(f'{lower_val}')
    lower_val = 2.3
    #     upper_val=2.5
    for i in range(0, 2):
        mstm_values = df_interpol[(df_interpol[feature_x] == lower_val)]['g']
        KRR_values = df_interpol[(df_interpol[feature_x] == lower_val)]['g_pred_KRR']
        NN_values = df_interpol[(df_interpol[feature_x] == lower_val)]['g_pred_NN']
        l1 = KRR_values.to_numpy() - mstm_values
        l2 = NN_values.to_numpy() - mstm_values
        data_KRR.append(l1)
        data_NN.append(l2)
        ticks.append(f'{lower_val}')
        lower_val = 2.5

    # Creating plot
    bpl = axs[2, 1].boxplot(data_KRR, labels=ticks, positions=np.array(range(len(data_KRR))) * 2.0 - 0.4, sym="")
    bpr = axs[2, 1].boxplot(data_NN, labels=ticks, positions=np.array(range(len(data_NN))) * 2.0 + 0.4, sym="")
    axs[2, 1].grid(linestyle="dashed")
    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')
    axs[2, 1].plot([], c='#D7191C')
    axs[2, 1].plot([], c='#2C7BB6')
    axs[2, 1].set_xticks(range(0, len(ticks) * 2, 2), ticks)
    axs[2, 1].set(xlabel='$D_{f}$')
    ######
    data_KRR = []
    data_NN = []
    ticks = []
    lower_val = 2.7
    #     upper_val=1.7
    mstm_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['g']
    KRR_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['g_pred_KRR']
    NN_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['g_pred_NN']
    l1 = KRR_values.to_numpy() - mstm_values
    l2 = NN_values.to_numpy() - mstm_values
    data_KRR.append(l1)
    data_NN.append(l2)
    ticks.append(f'{lower_val}')
    #     lower_val=2.9
    lower_val = 2.9
    for i in range(0, 1):
        mstm_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['g']
        KRR_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['g_pred_KRR']
        NN_values = df_extrapol[(df_extrapol[feature_x] == lower_val)]['g_pred_NN']
        l1 = KRR_values.to_numpy() - mstm_values
        l2 = NN_values.to_numpy() - mstm_values
        data_KRR.append(l1)
        data_NN.append(l2)
        ticks.append(f'{lower_val}')
    #         lower_val=2.3
    #         upper_val=2.9

    # Creating plot
    bpl = axs[2, 2].boxplot(data_KRR, labels=ticks, positions=np.array(range(len(data_KRR))) * 2.0 - 0.4, sym="")
    bpr = axs[2, 2].boxplot(data_NN, labels=ticks, positions=np.array(range(len(data_NN))) * 2.0 + 0.4, sym="")
    axs[2, 2].grid(linestyle="dashed")
    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')
    axs[2, 2].plot([], c='#D7191C')
    axs[2, 2].plot([], c='#2C7BB6')
    axs[2, 2].set_xticks(range(0, len(ticks) * 2, 2), ticks)
    axs[2, 2].set(xlabel='$D_{f}$')
    fig.tight_layout()
    fig.legend(loc='upper center', bbox_to_anchor=(1.05, 0.5))

    fig.savefig(f"boxplots/boxplot_df.pdf", format="pdf", bbox_inches="tight")




create_boxplots(df_random, df_extrapol, df_interpol, 'fractal_dimension')