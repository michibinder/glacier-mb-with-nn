from __future__ import print_function, division
import numpy as np
from netCDF4 import Dataset, num2date
import statsmodels.api as sm
import matplotlib.pyplot as plt
import network3
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU
# softmax plus log-likelihood cost is more common in modern image classification networks.

'input the mass balance data'
mb_taku = np.loadtxt('taku_mb.csv', delimiter=',')
M_dot_taku = mb_taku[:, 1]
years_taku = mb_taku[:, 0]

# plt.figure()
# plt.plot(years_taku, M_dot_taku, 'o')
# plt.xlabel('Years')
# plt.ylabel('m [mm/year]')
# plt.title('Mass balance')

'input the climate data'
clim = Dataset('taku_climate.nc', 'a')
# read the variables
clim_time = clim.variables['time'][:]  # days since 1801-01-01 00:00:00
clim_time_units = clim.variables['time'].units
clim_temp = clim.variables['temp'][:]  # in degC
clim_prec = np.sqrt(clim.variables['prcp'][:])  # in kg m-2S
# convert to dates
clim_dates = num2date(clim_time[:], units=clim_time_units, calendar='standard')

# Note, taku MB from 1946 - 2015
# climate data in Oct 1901 to Sep 2016
# use climate data from index 531 to 1370

'filter the temp and prec data to fit the MB idea'
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
# print(clim_dates[index_hydro[0]])
# print(clim_dates[index_hydro[-1]])

clim_dates_reshape = np.reshape(clim_dates[index_used], (70, 12))
clim_temp_reshape = np.reshape(clim_temp_f[index_used], (70, 12))
clim_prec_reshape = np.reshape(clim_prec_f[index_used], (70, 12))

td = [[],[]]
ed = [[],[]]

for i in range(0, clim_dates_reshape.shape[0]-10):
    item = np.append(clim_temp_reshape[i,:], clim_prec_reshape[i,:])
    item = list(np.matrix.transpose(item))
    td[0].append(item)
    td[1].append(M_dot_taku[i])


for i in range(clim_dates_reshape.shape[0]-11, clim_dates_reshape.shape[0]):
    item = np.append(clim_temp_reshape[i,:], clim_prec_reshape[i,:])
    item = list(np.matrix.transpose(item))
    ed[0].append(item)
    ed[1].append(M_dot_taku[i])

print(td)
# 70 entries with 24 entries respectively

mini_batch_size = 70
'''
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)
'''

net = Network([
    FullyConnectedLayer(n_in=24, n_out=12, activation_fn=ReLU),
    SoftmaxLayer(n_in=12, n_out=1)], mini_batch_size)

net = network2.Network([24, 12, 1], cost=network2.CrossEntropyCost)
#net.large_weight_initializer()
net.SGD(td, 30, 2, 0.1, lmbda = 5.0,evaluation_data=ed,monitor_evaluation_accuracy=True)

# training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
# training_results = [vectorized_result(y) for y in tr_d[1]]
# training_data = zip(training_inputs, training_results)

# plt.figure()
# plt.plot(clim_temp_ysum, M_dot_taku, 'o')
# plt.xlabel('Temp')
# plt.ylabel('Mdot')
# plt.title('Mdot vs Temperature')

# plt.figure()
# plt.plot(clim_prec_ysum, M_dot_taku, 'o')
# plt.xlabel('Prec')
# plt.ylabel('Mdot')
# plt.title('Mdot vs Precipitation')

# plt.figure()
# plt.plot(clim_prec_ysum, clim_temp_ysum, 'o')
# plt.xlabel('Prec')
# plt.ylabel('Temp')
# plt.title('Precipitation vs Temperature')

plt.show()
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# #net.large_weight_initializer()
# net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
#     monitor_evaluation_accuracy=True)