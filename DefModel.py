"""
This module allows for easy definition of the network based on the
activation function and size/shape dictionary

M. DiMario (2020)

"""
import tensorflow as tf
import numpy as np

def nn_model(x_data, shape_dict, init_sdev, afun, afun_max, bias_init, relu_alpha):

    L_input = x_data

    for elem_ind, elem in enumerate(shape_dict):
        init_sdev_xavier = np.sqrt(1/shape_dict[elem][0])
        Winit = tf.truncated_normal_initializer(mean=0.0, stddev=init_sdev_xavier)
        Binit = tf.constant(0.0, shape=(shape_dict[elem][1],))

        Wname = elem
        Bname = 'b' + elem[1:]

        W = tf.get_variable(name=Wname, shape=shape_dict[elem], initializer=Winit)
        B = tf.get_variable(name=Bname, initializer=Binit)

        if afun == 'sigmoid':
            L_output = tf.nn.sigmoid(tf.add(tf.matmul(L_input, W), B))
        elif afun == 'tanh':
            L_output = tf.nn.tanh(tf.add(tf.matmul(L_input, W), B))
        elif afun == 'relu':
            L_output = tf.nn.relu(tf.add(tf.matmul(L_input, W), B))
        elif afun == 'Lrelu':
            L_output = tf.nn.leaky_relu(tf.add(tf.matmul(L_input, W), B), alpha=relu_alpha)

        L_input = L_output

    if afun == 'sigmoid':
        f_out_init = tf.constant(2*afun_max, shape=(shape_dict[elem][1],))
        f_out = tf.get_variable(name='f_out', initializer=f_out_init)
        b_out_init = tf.constant(bias_init, shape=(shape_dict[elem][1],))
        b_out = tf.get_variable(name='b_out', initializer=b_out_init)

    elif afun == 'tanh':
        f_out_init = tf.constant(afun_max, shape=(shape_dict[elem][1],))
        f_out = tf.get_variable(name='f_out', initializer=f_out_init)
        b_out_init = tf.constant(bias_init, shape=(shape_dict[elem][1],))
        b_out = tf.get_variable(name='b_out', initializer=b_out_init)

    elif afun == 'relu':
        f_out_init = tf.constant(afun_max, shape=(shape_dict[elem][1],))
        f_out = tf.get_variable(name='f_out', initializer=f_out_init)
        b_out_init = tf.constant(bias_init, shape=(shape_dict[elem][1],))
        b_out = tf.get_variable(name='b_out', initializer=b_out_init)

    elif afun == 'Lrelu':
        f_out_init = tf.constant(afun_max, shape=(shape_dict[elem][1],))
        f_out = tf.get_variable(name='f_out', initializer=f_out_init)
        b_out_init = tf.constant(bias_init, shape=(shape_dict[elem][1],))
        b_out = tf.get_variable(name='b_out', initializer=b_out_init)

    model_out = tf.multiply(L_output, f_out)
    model_out = tf.add(model_out, b_out*[1, 0] + [0, b_out_init[1]], name='nn_output')

    return model_out
