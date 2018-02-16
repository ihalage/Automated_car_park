"""
Neural Network Model

"""

__all__ = (
    'final_training_model',
    'final_detect_model',
    'INPUT_IMAGE_SHAPE',
)

import tensorflow as tf
import functions


INPUT_IMAGE_SHAPE = (32, 256)

def weights(shape):
    weights = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weights)


def biases(shape):
    biases = tf.constant(0.1, shape=shape)
    return tf.Variable(biases)


def conv2d(x, W, stride=(1,1), padding ='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding=padding)


def max_pool(x, ksize=(2,2), stride = (2,2), padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                                    strides=[1, stride[0], stride[1], 1], padding=padding)


def avg_pool(x, ksize=(2,2), stride = (2,2), padding='SAME'):
    return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                                    strides=[1, stride[0], stride[1], 1], padding=padding)


"""

********Describe convolutional layers*********

Input image (256 x 32)
    |
4x4 Convolve with 48 feature maps
    |
2x2 max pool (128x16)
    |
4x4 convolve with 64 feature maps
    |
2x1 max pool (64x16)
    |
4x4 convolve with 128 feature maps
    |
2x2 average pool (32x8)

"""

def convolutional_layers():


    x = tf.placeholder(tf.float32, [None, None, None])

    # First layer
    W_conv1 = weights([4, 4, 1, 48])
    b_conv1 = biases([48])
    x_expanded = tf.expand_dims(x, 3)

    h_conv1 = tf.nn.relu(conv2d(x_expanded, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))

    #Second layer
    W_conv2 = weights([4, 4, 48, 64])
    b_conv2 = biases([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, ksize=(1, 2), stride=(1, 2))

    # Third layer
    W_conv3 = weights([4, 4, 64, 128])
    b_conv3 = biases([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = avg_pool(h_conv3, ksize=(2, 2), stride=(2, 2))

    return x, h_pool3, [W_conv1, b_conv1,
                        W_conv2, b_conv2,
                        W_conv3, b_conv3]

def final_training_model():

    """
    produces (7 * len(functions.CHARS)) vector(v) as the output.
    v[i * 3] being fired means, the ith character is C
    """

    x, conv_layer, conv_vars = convolutional_layers()

    # Densely connected layer
    W_fc1 = weights([32 * 8 * 128, 2048])
    b_fc1 = biases([2048])

    conv_layer_flat = tf.reshape(conv_layer, [-1, 32 * 8 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)

    # Output layer
    W_fc2 = weights([2048,7 * len(functions.CHARS)])
    b_fc2 = biases([7 * len(functions.CHARS)])

    y = tf.matmul(h_fc1, W_fc2) + b_fc2
    print y
    return (x, y, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2])
#final_training_model()
