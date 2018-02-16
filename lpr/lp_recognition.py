

__all__ = ('detect')


import collections
import itertools
import math
import sys
import time
import cv2
import numpy
import tensorflow as tf

import functions
import deep_net








def detect(im, param_vals):
    """
    Detect number plates in an image.

    :param im:
        Image to detect number plates in.

    :param param_vals:
        Model parameters to use. These are the parameters output by the `train`
        module.

    :returns:
        a 7,36 matrix giving the probability distributions of each letter.

    """

    # Load the model which detects number plates
    x, y, params = deep_net.final_training_model()

    # Execute the model at each scale.
    with tf.Session(config=tf.ConfigProto()) as sess:
        feed_dict = {x: numpy.stack([im])}
        feed_dict.update(dict(zip(params, param_vals)))
        y_val = sess.run(y, feed_dict=feed_dict)

    #finding the probabilities of each letter being present
    letter_probs = y_val.reshape(7,len(functions.CHARS))
    letter_probs = functions.softmax(letter_probs)

    return letter_probs

#Joining the letters with maximum probability
def letter_probs_to_code(letter_probs):
    joint_prob = 1
    for i in numpy.max(letter_probs, axis=1):
        print i
        joint_prob = joint_prob * i
    return "".join(functions.CHARS[i] for i in numpy.argmax(letter_probs, axis=1)), joint_prob


if __name__ == "__main__":
    start_time = time.time()
    im = cv2.imread(sys.argv[1])
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.

    f = numpy.load(sys.argv[2])
    param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]

    letter_probs = detect(im_gray, param_vals)


    code, confidence_level = letter_probs_to_code(letter_probs)
    if (code[0] == "Z"):
        print("Number Plate ->",code[1:])
        print("Confidence Level = ", confidence_level)
    else:
        print("Number Plate ->",code)
        print("Confidence Level = ", confidence_level)

    print("--- %s seconds ---" % (time.time() - start_time))
    ##color = (0.0, 255.0, 0.0)
    #cv2.rectangle(im, pt1, pt2, color)
"""
    cv2.putText(im,
                code,
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 0, 0),
                thickness=5)

    cv2.putText(im,
                code,
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 255),
                thickness=2)

    cv2.imwrite(sys.argv[3], im)
"""
