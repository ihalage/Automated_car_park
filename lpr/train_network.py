
"""
Code to train the neural network
"""

__all__ = (
    'train',
)


import functools
import glob
import itertools
import multiprocessing
import random
import sys
import time

import cv2
import numpy
import tensorflow as tf

import functions
import deep_net
import img_process

def code_to_vec(code):
    print code
    #To make an array where fired neuron val = 1.0 and all others 0
    def char_to_vec(c):
        y = numpy.zeros((len(functions.CHARS),))
        y[functions.CHARS.index(c)] = 1.0
        return y

    c = numpy.vstack([char_to_vec(c) for c in code])

    return c.flatten()


def read_data(img_glob):
    for fname in sorted(glob.glob(img_glob)):
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        _code = fname.split("/")[1]
        code = _code.split("_")[1]
        #p = fname.split("/")[1][17] == '1'
        yield im, code_to_vec(code)


def unzip(b):
    xs, ys = zip(*b)
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys


def read_batches(batch_size):
    #generator object
    g = img_process.img_code_init("/home/achintha/Academics/tensorflow/aa_lpr/test")
    def gen_vecs():
        for im, c in itertools.islice(g, batch_size):
            yield im, code_to_vec(c)

    while True:
        yield unzip(gen_vecs())


def get_loss(y, y_):
    # Calculate the loss from digits being incorrect.  Don't count loss from
    # digits that are in non-present plates.
    digits_loss = tf.nn.softmax_cross_entropy_with_logits(
                                          #logits=tf.reshape(y[:, 1:],
                                          logits=tf.reshape(y,
                                                     [-1, len(functions.CHARS)]),
                                          labels=tf.reshape(y_,
                                                     [-1, len(functions.CHARS)]))
    digits_loss = tf.reshape(digits_loss, [-1, 7])
    digits_loss = tf.reduce_sum(digits_loss, 1)
    #digits_loss *= (y_[:, 0] != 0)
    digits_loss = tf.reduce_sum(digits_loss)

    # Calculate the loss from presence indicator being wrong.
    #presence_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                                                          #logits=y[:, :1], labels=y_[:, :1])
    #presence_loss = 7 * tf.reduce_sum(presence_loss)

    return digits_loss


def train(learn_rate, report_steps, batch_size, initial_weights=None):
    """
    Train the network.

    The function operates interactively: Progress is reported on stdout, and
    training ceases upon `KeyboardInterrupt` at which point the learned weights
    are saved to `weights.npz`, and also returned.

    :param learn_rate:
        Learning rate to use.

    :param report_steps:
        Every `report_steps` batches a progress report is printed.

    :param batch_size:
        The size of the batches used for training.

    :param initial_weights:
        (Optional.) Weights to initialize the network with.

    :return:
        The learned network weights.

    """
    x, y, params = deep_net.final_training_model()

    y_ = tf.placeholder(tf.float32, [None, 7 * len(functions.CHARS)])

    digits_loss = get_loss(y, y_)
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(digits_loss)

    best = tf.argmax(tf.reshape(y, [-1, 7, len(functions.CHARS)]), 2)
    correct = tf.argmax(tf.reshape(y_, [-1, 7, len(functions.CHARS)]), 2)

    if initial_weights is not None:
        assert len(params) == len(initial_weights)
        assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]

    init = tf.global_variables_initializer()

    def vec_to_plate(v):
        return "".join(functions.CHARS[i] for i in v)


    def do_report():
        r = sess.run([best,
                      correct,
                      digits_loss],
                     feed_dict={x: test_xs, y_: test_ys})
        #num_correct = numpy.sum(numpy.logical_or(numpy.all(r[0] == r[1], axis=1)))

        r_short = (r[0][:190], r[1][:190])

        for b, c in zip(*r_short):
            print "{} {}".format(vec_to_plate(c), vec_to_plate(b))

        """********************"""

    def do_batch():
        sess.run(train_step,
                 feed_dict={x: batch_xs, y_: batch_ys})
        if batch_idx % report_steps == 0:
            do_report()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        if initial_weights is not None:
            sess.run(assign_ops)

        test_xs, test_ys = unzip(list(read_data("test/*.jpg"))[:50])

        try:
            last_batch_idx = 0
            last_batch_time = time.time()
            batch_iter = enumerate(read_batches(batch_size))
            for batch_idx, (batch_xs, batch_ys) in batch_iter:
                do_batch()
                if batch_idx % report_steps == 0:
                    batch_time = time.time()
                    if last_batch_idx != batch_idx:
                        print "time for 60 batches {}".format(
                            60 * (last_batch_time - batch_time) /
                                            (last_batch_idx - batch_idx))
                        last_batch_idx = batch_idx
                        last_batch_time = batch_time


        except KeyboardInterrupt:
            last_weights = [p.eval() for p in params]
            numpy.savez("weights.npz", *last_weights)
            return last_weights



if __name__ == "__main__":
    if len(sys.argv) > 1:
        f = numpy.load(sys.argv[1])
        initial_weights = [f[n] for n in sorted(f.files,
                                                key=lambda s: int(s[4:]))]
    else:
        initial_weights = None

    train(learn_rate=0.001,
          report_steps=20,
          batch_size=50,
          initial_weights=initial_weights)
