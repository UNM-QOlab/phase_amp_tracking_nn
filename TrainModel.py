"""
This is to train the neural network based on the monte carlo generated data
samples. The sample weights and network parameters can be tuned.

M. DiMario (2020)

"""
import matplotlib.pyplot as plt
import time
import numpy as np
import tensorflow as tf
from DefModel import nn_model

tf.reset_default_graph()

samples_per_iter = 5000
mpn_per_meas = 5.0
num_states = 4
PNR = 10
L = 10
num_states = 4
DE = 1.0
VIS = 0.997
DC = 0
exp_params = (DE, VIS, DC, num_states, L, PNR)
dmat_dim = num_states*(PNR+1)
in_dim = dmat_dim + 1
activation_function = 'Lrelu'

save_nn = False
folder_path = ''

num_outs = 2
num_iter = 10   #Number of training data text files
num_samples = num_iter*samples_per_iter #Total number of samples
p_train = 0.75
p_test = 1 - p_train
batch_frac = 1/100  #For batch training
ep_show_frac = 1/200    #How often we show the results
n_check = 2

#Learning parameters
lr0 = 50
lr = lr0*(10**-6) + 1e-20

mom = 0.80
decay = 0
drop_rate = 0.0
reg_param = 0.0
epochs = 2000
close = False #Close tf session or not
init_sdev = np.sqrt(0.02)

num_train = int(np.round(num_samples*p_train))
num_test = num_samples - num_train
n_vec = np.arange(num_train)
n_vec_test = np.arange(num_test)
batch_size = int(num_train*batch_frac)
batch_size_test = int(num_test*batch_frac)

#Initialize empty arrays for data storage
Wa = np.zeros([batch_size, num_outs])
pred_test = np.zeros([num_test, num_outs])
pred_train = np.zeros([num_test, num_outs])
mse_train = np.zeros([epochs, num_outs])
mse_test = np.zeros([epochs, num_outs])

x_data = np.zeros([num_samples, in_dim])
y_data = np.zeros([num_samples, num_outs])
w = np.ones([num_samples, num_outs])

mpn_test = np.zeros([epochs, n_check])
mpn_check = np.zeros([epochs, n_check])
mpn_error = np.zeros([epochs, n_check])

##############################################################################################################################################################

#Load data samples
path_id = ''
load_path = folder_path + 'TrainingData/' + path_id + '/'

for i in range(num_iter):
    print('Loading samples data : ' + str(i))
    iH = samples_per_iter*(i+1)
    iL = samples_per_iter*i
    z0 = np.zeros([samples_per_iter, dmat_dim + 1 + 2 + 4])
    z0 = np.loadtxt(load_path + 'Sample' + str(i) + '.txt', delimiter=',   ')

    z_dmat = np.zeros([samples_per_iter, dmat_dim])
    z_dmat_norm = np.zeros([samples_per_iter, dmat_dim])
    for k in range(num_states):
        low = (k*(PNR+1))
        up = ((k+1)*(PNR+1))
        z_dmat[:, low:up] = z0[:, low:up]
        z_dmat_norm[:, low:up] = z_dmat[:, low:up]/(np.sum(1e-100 + z_dmat[:, low:up], axis=1)[:, np.newaxis])

    x_data[iL:iH, 0:dmat_dim] = 2*z_dmat_norm - 1
    x_data[iL:iH, dmat_dim] = z0[:, dmat_dim]/10 - 1

    y_data[iL:iH, 0] = z0[:, dmat_dim+1]/(2*int(ph_noise)/100)
    y_data[iL:iH, 1] = (z0[:, dmat_dim+2]/10 - 1)

    w[iL:iH, 0] = np.exp(-(z0[:, dmat_dim] - z0[:, dmat_dim+2])**2 / (2)) + 0.1
    w[iL:iH, 1] = np.exp(-(z0[:, dmat_dim] - z0[:, dmat_dim+2])**2 / (2)) + 0.1

afun_max = 1.5
bias_init = [1.75, 1.75]
relu_alpha = 0.1

arr = np.arange(num_samples)
np.random.shuffle(arr)
train_pts = arr[:num_train]
test_pts = arr[num_train:]

x_train = x_data[train_pts, :]
x_test = x_data[test_pts, :]

y_train = y_data[train_pts, :]
y_test = y_data[test_pts, :]

w_train = w[train_pts, :]
w_test = w[test_pts, :]

##############################################################################################################################################################

tf.reset_default_graph()

x = tf.placeholder('float', name='nn_input')
actual = tf.placeholder('float')
weight = tf.placeholder('float')
t = tf.placeholder('float')
learn_rate = tf.placeholder('float')

shape_vec = np.array([0, 32, 32, 32, 32, 16, 16, 8, 8, 0])
shape_vec[0] = in_dim
shape_vec[-1] = num_outs

shape_dict = {}
for i in range(len(shape_vec)-1):
    name = 'w' + str(i)
    shape_dict[name] = (shape_vec[i], shape_vec[i+1])
shape_dict['w' + str(i+1)] = (shape_vec[i+1], num_outs)

output = nn_model(x,
                  shape_dict,
                  init_sdev,
                  activation_function,
                  afun_max,
                  bias_init,
                  relu_alpha)

# define cost functions
mse_cost = tf.losses.mean_squared_error(actual, output, weight)
reg_loss = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

loss_fun = mse_cost #Can add reg_loss if needed

# create an optimzier with a learning rate and cost function
optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=mom).minimize(loss_fun)
optimizer_sd = 'RMSprop'

##############################################################################################################################################################

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

# start tf session
sess = tf.Session()

#initialize variables
sess.run(init_op)

total_params = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    print('Variable ' + variable.name + ' has shape : ' + str(shape))
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_params += variable_parameters

for ep in range(epochs):
    np.random.shuffle(n_vec)
    np.random.shuffle(n_vec_test)

    num_batches = int(num_train/batch_size)

    for b in range(num_batches):
        indS = b*batch_size
        indE = (b+1)*batch_size
        ind = n_vec[indS : indE];

        indS_test = b*batch_size_test
        indE_test = (b+1)*batch_size_test
        ind_test = n_vec_test[indS_test : indE_test];

        X = x_train[ind, :].reshape(batch_size, in_dim)
        Y = y_train[ind, :].reshape(batch_size, num_outs)

        W = w_train[ind, :].reshape(batch_size, num_outs)

        #Run the training
        _, c = sess.run([optimizer, loss_fun], feed_dict={x: X, actual: Y, weight: W})

    #Check the training and validation sets
    pred_train = sess.run(output, feed_dict={x: x_train})
    mse_train[ep, :] = np.mean((pred_train - y_train)**2, axis=0)

    pred_test = sess.run(output, feed_dict={x: x_test})
    mse_test[ep, :] = np.mean((pred_test - y_test)**2, axis=0)

    if np.mod(ep, np.round(epochs*ep_show_frac)) == 0:
        pred_train = sess.run(output, feed_dict={x: x_train})
        pred_test = sess.run(output, feed_dict={x: x_test})

        plt.figure(0)
        for no in range(num_outs):
            plt.subplot(1, num_outs, no+1)
            plt.hexbin(y_train[:,no], pred_train[:, no] - y_train[:, no], bins='log')
            plt.plot(y_train[:, no], y_train[:, no] - y_train[:, no], 'k')
        plt.show()

        plt.figure(1)
        for no in range(num_outs):
            plt.subplot(1, num_outs, no+1)
            plt.hexbin(y_test[:,no], pred_test[:, no] - y_test[:, no], bins='log')
            plt.plot(y_test[:, no], y_test[:, no] - y_test[:, no], 'k')
        plt.show()

        plt.figure(2)
        for no in range(num_outs):
            plt.subplot(1, num_outs, no+1)
            plt.semilogy(mse_train[0:ep+1, no])
            plt.semilogy(mse_test[0:ep+1, no])
        plt.show()

        print('Epoch', ep, 'completed out of', epochs)

nn_path = folder_path + 'TrainedModels/' + path_id + ''
save_path = saver.save(sess, nn_path + 'Model_' + path_id + '')
save_dict = {'mpn': mpn_per_meas,
             'training_params': path_id,
             'learning rate': lr0,
             'activation function': activation_function,
             'momentum': mom,
             'decay': decay,
             'drop rate': drop_rate,
             'reg param': reg_param,
             'batch fraction': batch_frac,
             'epochs': epochs,
             'initial sdev': init_sdev,
             'total_parameters': total_params,
             'total_train_pts': num_samples,
             'train_frac': p_train,
             'final train mse': mse_train[-1, :],
             'final test mse': mse_test[-1, :]}

if save_nn:
    np.save(nn_path + 'SaveDictionary.npy', save_dict)

print('Finished NN training')

all_params = np.array([])
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    s = np.prod(shape)
    all_params = np.append(all_params, np.ndarray.flatten(sess.run(variable)))

if close:
    sess.close()
    tf.reset_default_graph()
