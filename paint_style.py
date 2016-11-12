import argparse
import time
import scipy.misc
import numpy as np
import style as st

# Default values
_default_learning_rate = 1e1
_default_iterations = 1000
_default_content_weight = 1e0
_default_style_weight = 1e3
_default_check_per_iteration = 100
_default_vgg = 'imagenet-vgg-verydeep-19.mat'
_default_output = 'output.jpg'


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--content', dest = 'content', help = 'Input content image', required = True)
    parser.add_argument('-s', '--style', dest = 'style', help = 'Style image', required = True)
    parser.add_argument('-o', '--output', dest = 'output', help = 'Output image', default = _default_output)
    parser.add_argument('--vgg', dest = 'vgg', help = 'Path to pretrained vgg19 network', default = _default_vgg)
    parser.add_argument('--content-weight', type = float, dest = 'content_weight', help = 'Weight for content (input) image', default = _default_content_weight)
    parser.add_argument('--style-weight', type = float, dest = 'style_weight', help = 'Weight for style image', default = _default_style_weight)
    parser.add_argument('--check-per-iteration', type = int, dest = 'check_per_iteration', help = 'Frequency of checking current loss', default = _default_check_per_iteration)
    parser.add_argument('-a', '--learning-rate', type = float, dest = 'learning_rate', help = 'Learning rate for neural network', default = _default_learning_rate)
    parser.add_argument('-i', '--iterations', type = int, dest = 'iterations', help = 'Max iterations', default = _default_iterations)


    return parser.parse_args()

def run(arguments):
    # Load images
    content = scipy.misc.imread(arguments.content).astype(np.float)
    style = scipy.misc.imread(arguments.style).astype(np.float)
    # Resize style image so it is the same size as content
    style = scipy.misc.imresize(style, content.shape[1] / style.shape[1])

    print("Running neural style algorithm. Output will be stored in: " + arguments.output)

    result = st.convert_style(
                net_path = arguments.vgg,
                content = content, 
                style = style, 
                iterations = arguments.iterations, 
                content_weight = arguments.content_weight, 
                style_weight = arguments.style_weight, 
                learning_rate = arguments.learning_rate,
                check_per_iteration = arguments.check_per_iteration
            )

    # Convert result from float image to uint8 image
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Save image
    scipy.misc.imsave(arguments.output, result)

if __name__ == '__main__':
    t = time.time()

    run(parse_arguments())

    print("Processing complete. Time elapsed:")
    second = time.time() - t
    hour = int(second / 3600)
    second -= hour * 3600
    minute = int(second / 60)
    second -= minute * 60
    print("{0} Hours, {1} Minutes and {2} Seconds".format(hour, minute, second))