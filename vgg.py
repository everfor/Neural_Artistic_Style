import tensorflow as tf
import numpy as np
import scipy.io

vgg_layers = [
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
]

vgg_layer_types = [
    'conv', 'relu', 'conv', 'relu', 'pool',
    'conv', 'relu', 'conv', 'relu', 'pool',
    'conv', 'relu', 'conv', 'relu', 'conv', 'relu', 'conv', 'relu', 'pool',
    'conv', 'relu', 'conv', 'relu', 'conv', 'relu', 'conv', 'relu', 'pool',
    'conv', 'relu', 'conv', 'relu', 'conv', 'relu', 'conv', 'relu'
]

def build_net(path_network, input_image):
    # Load pretrained convnet
    pretrained_net = scipy.io.loadmat(path_network)
    # Mean of input pixels - used to normalize input images
    mean = np.mean(pretrained_net['normalization'][0][0][0], axis = (0, 1))
    layers = pretrained_net['layers'][0]

    convnet = {}
    current = input_image
    for i, name in enumerate(vgg_layers):
        if vgg_layer_types[i] == 'conv':
            # Convolution layer
            kernel, bias = layers[i][0][0][0][0]
            # (width, height, in_channels, out_channels) -> (height, width, in_channels, out_channels)
            kernels = np.transpose(kernel, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            conv = tf.nn.conv2d(current, tf.constant(kernel), strides = (1, 1, 1, 1), padding = 'SAME')
            current = tf.nn.bias_add(conv, bias)
        elif vgg_layer_types[i] == 'relu':
            # Relu layer
            current = tf.nn.relu(current)
        elif vgg_layer_types[i] == 'pool':
            # Pool layer
            current = tf.nn.avg_pool(current, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = 'SAME')
        convnet[name] = current

    return convnet, mean


def pre_process_image(image, mean_pixel):
    return image - mean_pixel


def restore_image(image, mean_pixel):
    return image + mean_pixel
