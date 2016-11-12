import vgg
import tensorflow as tf
import numpy as np

content_layer = 'relu4_2'
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

def convert_style(net_path, content, style, iterations, content_weight, style_weight, learning_rate, check_per_iteration):
    print("Total iterations: {0}".format(iterations))
    # Store shapes of both images
    content_shape = (1,) + content.shape
    style_shape = (1,) + style.shape

    # Features
    content_features = {}
    style_features = {}

    # Extract features for content
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as session:
        # Build convnet in tensorflow
        image = tf.placeholder('float', shape = content_shape)
        net, mean = vgg.build_net(net_path, image)
        # Extract features
        preprocessed_content = np.array([vgg.pre_process_image(content, mean)])
        content_features[content_layer] = net[content_layer].eval(
            feed_dict = {image: preprocessed_content})

    print("Content feature extracted")

    # Extract features for style
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as session:
        # Build convnet
        image = tf.placeholder('float', shape = style_shape)
        net, _ = vgg.build_net(net_path, image)
        # Extract style features
        preprocessed_style = np.array([vgg.pre_process_image(style, mean)])
        for layer in style_layers:
            layer_features = net[layer].eval(
                feed_dict = {image: preprocessed_style})
            style_features[layer] = layer_features

    print("Style feature extracted")

    # Reconstruct image through backprogpagation
    g = tf.Graph()
    with g.as_default():
        # Random generated image as initial state
        image = tf.Variable(tf.random_normal(content_shape) * 0.256)
        # Build convnet for backprogpagation
        net, _ = vgg.build_net(net_path, image)

        # Calculate content loss
        content_loss =  2 * tf.nn.l2_loss(net[content_layer] - content_features[content_layer]) / content_features[content_layer].size

        # Calculate style loss
        style_loss = 0
        for layer in style_layers:
            # Gram of original convnet layers
            net_layer = net[layer]
            _, height, width, channels = map(lambda i: i.value, net_layer.get_shape())
            net_size = height * width * channels
            net_features = tf.reshape(net_layer, (-1, channels))
            net_gram = tf.matmul(tf.transpose(net_features), net_features) / net_size
            # Gram of style
            style_layer = style_features[layer]
            style_layer_features = np.reshape(style_layer, (-1, style_layer.shape[3]))
            style_gram = np.matmul(style_layer_features.T, style_layer_features) / style_layer_features.size
            # Style loss of current layer
            style_loss += 2 * tf.nn.l2_loss(net_gram - style_gram) / style_gram.size

        # Total loss
        total_loss = content_weight * content_loss + style_weight * style_loss

        # Train step
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        # Optimize
        least_loss = float('inf')
        best_img = None
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())

            for i in range(iterations):
                train_step.run()
                print("Iteration {0}/{1} complete.".format(i + 1, iterations))

                # If a check or last iteration is reached
                # Check if current image produces the least loss
                if i % check_per_iteration == 0 or i == iterations - 1:
                    curr_loss = total_loss.eval()
                    if (curr_loss < least_loss):
                        least_loss = curr_loss
                        best_img = image.eval()

        return vgg.restore_image(best_img.reshape(content_shape[1:]), mean)
