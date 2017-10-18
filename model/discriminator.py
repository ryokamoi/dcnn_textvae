import sys

import numpy as np
import tensorflow as tf

sys.path.append("../")

from config import FLAGS


class Discriminator(object):
    def __init__(self, encoder_rnn_output, temperature, is_training=True, ru=False):
        with tf.variable_scope("Discriminator_input"):
            self.encoder_rnn_output = encoder_rnn_output
            self.temperature = temperature

            self.is_training = is_training

        with tf.variable_scope("discriminator_linear1"):
            discriminator_W1 = tf.get_variable(name="discriminator_W1",
                                              shape=(FLAGS.RNN_SIZE, 100),
                                              dtype=tf.float32,
                                              initializer=tf.random_normal_initializer(stddev=0.1))
            discriminator_b1 = tf.get_variable(name="discriminator_b1",
                                              shape=(100),
                                              dtype=tf.float32)

        with tf.variable_scope("discriminator_linear2"):
            discriminator_W2 = tf.get_variable(name="discriminator_W2",
                                              shape=(100, FLAGS.LABEL_CLASS),
                                              dtype=tf.float32,
                                              initializer=tf.random_normal_initializer(stddev=0.1))
            discriminator_b2 = tf.get_variable(name="discriminator_b2",
                                              shape=(FLAGS.LABEL_CLASS),
                                              dtype=tf.float32)

        with tf.name_scope("hidden"):
            h = tf.nn.relu(tf.matmul(self.encoder_rnn_output, discriminator_W1) + discriminator_b1)

        with tf.name_scope("discriminator_output"):
            self.discriminator_logits = tf.matmul(h, discriminator_W2) + discriminator_b2
            self.discriminator_predict = tf.stop_gradient(tf.argmax(self.discriminator_logits, 1))
            self.discriminator_prob = tf.nn.softmax(self.discriminator_logits, name="discriminator_softmax")

        with tf.name_scope("sampling"):
            # unlabeled
            self.discriminator_sampling_onehot = self.gumbel_softmax(self.discriminator_logits, self.temperature)


    def gumbel_softmax(self, logits, temperature, dim=-1):
        u = tf.random_uniform((FLAGS.BATCH_SIZE, FLAGS.LABEL_CLASS), minval=np.finfo(np.float32).tiny)
        g = - tf.log(-tf.log(u))
        onehot = tf.nn.softmax(tf.div((logits+g), temperature),
                               dim=dim,
                               name="discriminator_gumbel_softmax")

        return onehot
