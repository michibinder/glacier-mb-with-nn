import numpy as np
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

## Data Preprocessing
datapath = 'aws_data_zhadang_UTC+6.csv' #adapt datapath

pd.options.display.max_rows = 20 #number of displayed rows

alldataset = pd.read_csv(datapath,index_col=0, parse_dates=True)

dropAngle           = True
dropWindSpeed       = False
dropWindDirection   = False
dropRadiationFluxes = False
dropSurfaceTemp     = False
dropRH              = False
dropPressure        = False

# SURFTEMP, PRESSURE, RH
if (dropAngle):
    alldataset.drop(['ANGLE'], axis=1, inplace = True)
if (dropWindSpeed):
    alldataset.drop(['WINDSPEED'], axis=1, inplace = True)
if (dropWindDirection):
    alldataset.drop(['WINDDIR'], axis=1, inplace = True)
if (dropRadiationFluxes):
    alldataset.drop(columns=['NETRAD', 'SWIN', 'SWOUT',], axis=1, inplace = True)
if (dropSurfaceTemp):
    alldataset.drop(['SURFTEMP'], axis=1, inplace = True)
if (dropRH):
    alldataset.drop(['RH'], axis=1, inplace = True)
if (dropPressure):
    alldataset.drop(['PRESSURE'], axis=1, inplace = True)

# Restrict data 
# dataset = dataset.iloc[0:500]
# dataset = alldataset['2012-05-15':'2012-08-31'] # only one ablation season
dataset = alldataset
print('.....Data loaded and restricted')

# Calculate and adjust delta SR50
dataset['dSR50'] = dataset['SR50'].diff() * 1000 # scaling
print('.....dSR50 determined')

# Drop Nans and infinity numbers
N_data_bef = len(dataset)
dataset = dataset[np.isfinite(dataset['dSR50'])]
N_data = len(dataset)
print('.....NaNs dropped, old: {}, new: {}'.format(N_data_bef, N_data))

# Split the dataset for training and testing (validation?)
# frac = int(N_data*0.8)
# train_dataset = dataset.iloc[0:frac]
train_dataset = dataset.sample(frac=0.3, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
print('.....Data splitted into Training and Testing')

# Pop the label, the target value
train_labels = train_dataset.pop('dSR50')
test_labels = test_dataset.pop('dSR50')
print('.....Target Value defined')


# Normalize the dataset
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
print('.....Statistics calculated')
# print(train_stats)

def norm(x):
    # Adjust function for non-negative data (precipitation, ...)?
    x = (x - train_stats['mean']) / train_stats['std']
    x.fillna(value = 0.0, inplace = True)
    return x

# this is what the ANN understands
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
print('.....Data normalized')
# print(normed_train_data)
# print(normed_test_data)

# Artifical Neural Network (Tensor Flow)
def build_model():
    model = keras.Sequential([
      layers.Dense(8, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
      # layers.Dense(15, activation=tf.nn.relu),
      # layers.Dense(30, activation=tf.nn.relu),
      layers.Dense(1)
    ])

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

EPOCHS = 100
print('.....Neural network initialized, Epochs: {}'.format(EPOCHS))
# history = model.fit(
#   normed_train_data, train_labels,
#   epochs=EPOCHS, validation_split=0.2, verbose=0,
#   callbacks=[PrintDot()])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, batch_size=10, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [SR50]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$SR50^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.legend()
    plt.show()


plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("\nTesting set Mean Abs Error: {:5.2f} SR50".format(mae))

# now use the whole input data to predict
alldataset = dataset
alldatalabels = alldataset.pop('dSR50')
normed_alldata = norm(dataset) # norm function is based on training set

predictions = model.predict(normed_alldata).flatten()

# some postprocesing
print("Comparative Stats")
# print("R2  of Neural Network : %.02f" % (R2(dataset['dSR50'], test_predictions)))
print("R2  of Neural Network : %.02f" % (R2(alldatalabels, predictions)))
print("STD of Neural Network : %.02f" % (np.std(alldatalabels - predictions)))

dataset['dSR50_pred'] = predictions
dataset['SR50_pred'] = np.cumsum(dataset['dSR50_pred'] / 1000) + dataset['SR50'].iloc[0]
# dataset['SR50_pred'] = np.cumsum(alldatalabels) Cumsum works fine
dataset['SR50'].plot()
dataset['SR50_pred'].plot()

# plt.figure()
# plt.plot(years_taku_use, M_dot_taku_use, '-ob', label='Data')
# plt.plot(years_taku_use, m_model_dd, '-or', label='T/P index model')
# plt.plot(years_taku_use, test_predictions, '-og', label='NN model')
# plt.xlabel('Years')
# plt.ylabel('m [mm/year]')
# plt.title('Mass balance')
# plt.legend()
# plt.show()