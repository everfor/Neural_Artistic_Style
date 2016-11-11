from vgg import VggConvnet
import scipy.misc
import tensorflow as tf
import numpy as np

if __name__ == '__main__':

    content = scipy.misc.imread("starry_night.jpeg")

    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as session:
        shape = (1, ) + content.shape
        print(shape)
        empty = tf.placeholder('float', shape = shape)
        net, mean = VggConvnet.build_net("imagenet-vgg-verydeep-19.mat", empty)
        print(content.shape)
        print(mean.shape)
        preprocessed = np.array([VggConvnet.pre_process_image(content, mean)])
        print(preprocessed.shape)
        net['relu4_2'].eval(feed_dict = {empty: preprocessed})