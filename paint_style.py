import vgg
import style
import scipy.misc
import tensorflow as tf
import numpy as np

if __name__ == '__main__':

    content_img = scipy.misc.imread("content.jpeg")
    style_img = scipy.misc.imread("style.jpeg")
    net = "imagenet-vgg-verydeep-19.mat"

    result = style.convert_style(net, None, content_img, style_img, 100, 5e0, 1e2, 1e1)

    scipy.misc.imsave("result.png", result)