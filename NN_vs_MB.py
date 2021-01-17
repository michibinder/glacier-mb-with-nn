import numpy as np
from netCDF4 import Dataset, num2date
import statsmodels.api as sm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd


def R2(data, model):
    data_mean = np.mean(data)
    SS_tot = np.sum((data - data_mean)**2.)
    SS_res = np.sum((data - model)**2.)
    r2 = 1 - SS_res/SS_tot
    return r2

'''
Temperatures are set to 0 when below 0 -> no need for NN
already reaches 50% variability with individual temperatures and prec
to further understand the Network use own network to analyze trained weights?
'''

# input the mass balance data
mb_taku = np.loadtxt('taku_mb.csv', delimiter=',')
M_dot_taku = mb_taku[:, 1]
years_taku = mb_taku[:, 0]

'input the climate data'
clim = Dataset('taku_climate.nc', 'a')
# clim = Dataset('RGI60-01.01390/climate_monthly.nc', 'a')
# read the variables
clim_time = clim.variables['time'][:]  # days since 1801-01-01 00:00:00
clim_time_units = clim.variables['time'].units
clim_temp = clim.variables['temp'][:]  # in degC
clim_prec = clim.variables['prcp'][:]  # in kg m-2S
# convert to dates
clim_dates = num2date(clim_time[:], units=clim_time_units, calendar='standard')

# Note, taku MB from 1946 - 2015
# climate data in Oct 1901 to Sep 2016
# use climate data from index 531 to 1370

# filter the temp and prec data to fit the MB idea
TMP_melt = 0.0  # degC
P_TMP_snow = 7.0  # degC

# clim_temp_f = np.maximum(clim_temp-TMP_melt, 0)
clim_temp_f = np.copy(clim_temp)
clim_temp_f[clim_temp < TMP_melt] = 0
clim_prec_f = np.copy(clim_prec)
clim_prec_f[clim_temp > P_TMP_snow] = 0

# reshape and sum climate variables
index_calendar = np.arange(531, 1371)
index_hydro = np.arange(528, 1368)
index_used = index_hydro
len_index_used = len(index_used)

clim_dates_reshape = np.reshape(clim_dates[index_used], (70, 12))
clim_temp_reshape = np.reshape(clim_temp_f[index_used], (70, 12))
clim_prec_reshape = np.reshape(clim_prec_f[index_used], (70, 12))

len_t_used = np.count_nonzero(clim_temp_f[index_used])
len_p_used = np.count_nonzero(clim_prec_f[index_used])

print("Only %.02f percent of tmp data used" % (len_t_used/len_index_used*100))
print("Only %.02f percent of pre data used" % (len_p_used/len_index_used*100))

clim_temp_ysum = np.sum(clim_temp_reshape, axis=1)
clim_prec_ysum = np.sum(clim_prec_reshape, axis=1)

# up to 70 years
N1 = 0
N2 = 70
print(mb_taku[N1:N2, 0])
clim_temp_ysum_use = clim_temp_ysum[N1:N2]
clim_prec_ysum_use = clim_prec_ysum[N1:N2]
M_dot_taku_use = M_dot_taku[N1:N2]
years_taku_use = years_taku[N1:N2]
# the independent variables namely Prec and Temp
X0 = np.column_stack((clim_prec_ysum_use, clim_temp_ysum_use))
# add constant for intercept
X = sm.add_constant(X0)

sm_massbal = sm.OLS(M_dot_taku_use, X).fit()
print(sm_massbal.summary())

beta_vals = sm_massbal.params

m_model_dd = beta_vals[0] + [1]*clim_prec_ysum_use + beta_vals[2]*clim_temp_ysum_use

# now comes the neural network
# convert to pandas dataframe

# dataset = pd.DataFrame({'MB': M_dot_taku_use,
#                         'TMP': clim_temp_ysum_use,
#                         'PRE': clim_prec_ysum_use})
print(clim_temp_reshape)
t_cols = ['T_JAN', 'T_FEB', 'T_MAR', 'T_APR', 'T_MAY', 'T_JUN', 'T_JUL', 'T_AUG', 'T_SEP', 'T_OCT', 'T_NOV', 'T_DEZ',
          'P_JAN', 'P_FEB', 'P_MAR', 'P_APR', 'P_MAY', 'P_JUN', 'P_JUL', 'P_AUG', 'P_SEP', 'P_OCT', 'P_NOV', 'P_DEZ']
dMatrix = np.column_stack([clim_temp_reshape, clim_prec_reshape])
dataset = pd.DataFrame(data = dMatrix, columns = t_cols)
dataset['MB'] = M_dot_taku_use
print(dataset)

# split the dataset to train and test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
#
# Pop the label, the target value
train_labels = train_dataset.pop('MB')
test_labels = test_dataset.pop('MB')

# normalize the dataset
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()


def norm(x):
    x = (x - train_stats['mean']) / train_stats['std']
    x.fillna(value = 0.0, inplace = True)
    return x

# this is what the ANN understands
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
print(normed_train_data)
print(normed_test_data)


def build_model():
    model = keras.Sequential([
      layers.Dense(12, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
      #layers.Dense(12, activation=tf.nn.relu),
      layers.Dense(1)
    ])

    # model = keras.Sequential([
    #   layers.Dense(12, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    #   layers.Dense(1)
    # ])

    # model = keras.Sequential([
    #   layers.Dense(12, activation=tf.nn.sigmoid, input_shape=[len(train_dataset.keys())]),
    #   layers.Dense(12, activation=tf.nn.sigmoid),
    #   layers.Dense(1)
    # ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


model = build_model()


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


EPOCHS = 4000

# history = model.fit(
#   normed_train_data, train_labels,
#   epochs=EPOCHS, validation_split=0.2, verbose=0,
#   callbacks=[PrintDot()])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MB]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MB^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.legend()
    plt.show()


plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("\nTesting set Mean Abs Error: {:5.2f} MB".format(mae))

# now use the whole input data to predict
alldataset = dataset
alldatalabels = alldataset.pop('MB')
normed_alldata = norm(alldataset)

test_predictions = model.predict(normed_alldata).flatten()

# some postprocesing
print("Comparative Stats")
print("R2  of T/P index model: %.02f" % (R2(M_dot_taku_use, m_model_dd)))
print("STD of T/P index model: %.02f" % (np.std(M_dot_taku_use - m_model_dd)))
print("R2  of Neural Network : %.02f" % (R2(M_dot_taku_use, test_predictions)))
print("STD of Neural Network : %.02f" % (np.std(M_dot_taku_use - test_predictions)))

plt.figure()
plt.plot(years_taku_use, M_dot_taku_use, '-ob', label='Data')
plt.plot(years_taku_use, m_model_dd, '-or', label='T/P index model')
plt.plot(years_taku_use, test_predictions, '-og', label='NN model')
plt.xlabel('Years')
plt.ylabel('m [mm/year]')
plt.title('Mass balance')
plt.legend()
plt.show()