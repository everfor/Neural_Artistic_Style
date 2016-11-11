import tensorflow as tf
import numpy as np
import scipy.io

class VggConvnet:

    # VGG19 convnet structure without fc layers
    vgg_layers = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool4',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
    ]

    @classmethod
    def build_net(this, path_to_network, input_image):
        pretrained_net = scipy.io.loadmat(path_to_network)
        mean = np.mean(pretrained_net['normalization'][0][0][0], axis = (0, 1))
        layers = pretrained_net['layers'][0]

        convnet = {}
        current = input_image

        for i, name in enumerate(VggConvnet.vgg_layers):
            # Get layer type
            layer_type = name[:4]
            print(name)

            if name == 'conv':
                # Convolution layer
                kernel, bias = weights[i][0][0][0][0]
                # (width, height, in_channels, out_channels) -> (height, width, in_channels, out_channels)
                kernel = np.transpose(kernel, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                # Create conv layer
                conv = tf.nn.conv2d(current, tf.constant(kernel), strides = (1, 1, 1, 1), padding = 'SAME')
                currnet = tf.nn.bias_add(conv, bias)
            elif name == 'relu':
                # Relu layer
                current = tf.nn.relu(current)
            else:
                # Pool layer
                current = tf.nn.avg_pool(current, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = 'SAME')

            convnet[name] = current


        return convnet, mean

    @staticmethod
    def pre_process_image(image, mean):
        return image - mean

    @staticmethod
    def restore_image(image, mean):
        return image + mean
