import sys

import tensorflow as tf

sys.path.append("../")

from config import FLAGS


class Encoder_cvae(object):
    def __init__(self, embedding, encoder_input_list,
                 is_training=True, ru=False):
        with tf.name_scope("encoder_input"):
            self.embedding = embedding
            self.encoder_input_list = encoder_input_list

            self.is_training = is_training

        with tf.variable_scope("encoder_rnn"):
            with tf.variable_scope("rnn_input_weight"):
                self.rnn_input_W = tf.get_variable(name="rnn_input_W",
                                                   shape=(FLAGS.EMBED_SIZE, FLAGS.RNN_SIZE),
                                                   dtype=tf.float32,
                                                   initializer=tf.random_normal_initializer(stddev=0.1))
                self.rnn_input_b = tf.get_variable(name="rnn_input_b",
                                                   shape=(FLAGS.RNN_SIZE),
                                                   dtype=tf.float32)

            with tf.variable_scope("encoder_rnn"):
                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(FLAGS.RNN_SIZE)

                if self.is_training:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                         output_keep_prob=FLAGS.ENCODER_DROPOUT_KEEP)

                self.cell = tf.contrib.rnn.MultiRNNCell([cell] * FLAGS.RNN_NUM)

                self.init_states = [cell.zero_state(FLAGS.BATCH_SIZE, tf.float32)
                                    for _ in range(FLAGS.RNN_NUM)]
                self.states = [tf.placeholder(tf.float32,
                                              (FLAGS.BATCH_SIZE),
                                              name="state")
                               for _ in range(FLAGS.RNN_NUM)]

        with tf.name_scope("encoder_rnn_output"):
            self.encoder_rnn_output = self.rnn_train_predict()


    # input text from dataset
    def rnn_train_predict(self):
        pred = []
        state = self.init_states
        for i in range(FLAGS.SEQ_LEN):
            with tf.name_scope("encoder_input_embedding"):
                encoder_input = self.encoder_input_list[i]
                encoder_input_embedding = tf.nn.embedding_lookup(self.embedding, encoder_input)
                assert encoder_input_embedding.shape == (FLAGS.BATCH_SIZE,
                                                         FLAGS.EMBED_SIZE)

            with tf.name_scope("rnn_input"):
                rnn_input = tf.nn.relu(tf.matmul(encoder_input_embedding, self.rnn_input_W) + self.rnn_input_b)
                assert rnn_input.shape == (FLAGS.BATCH_SIZE, FLAGS.RNN_SIZE)

            with tf.name_scope("rnn_predict"):
                step_pred, state = self.cell(rnn_input, state)
                assert state[-1][1].shape == (FLAGS.BATCH_SIZE, FLAGS.RNN_SIZE)
                assert step_pred.shape == (FLAGS.BATCH_SIZE, FLAGS.RNN_SIZE)

            pred.append(step_pred)

        return state[-1][1] # last hidden state


class Sampler(object):
    def __init__(self, encoder_rnn_output, label_onehot, is_training=True):
        self.encoder_rnn_output = encoder_rnn_output
        self.label_onehot = label_onehot

        self.is_training = is_training


        with tf.variable_scope("encoder_linear1"):
            context_to_hidden_W = tf.get_variable(name="context_to_hidden_W",
                                                  shape=[FLAGS.RNN_SIZE + FLAGS.LABEL_CLASS,
                                                         100],
                                                  dtype=tf.float32,
                                                  initializer=tf.random_normal_initializer(stddev=0.1))

            context_to_hidden_b = tf.get_variable(name="context_to_hidden_b",
                                                  shape=[100],
                                                  dtype=tf.float32)


        with tf.variable_scope("encoder_linear2"):
            context_to_mu_W = tf.get_variable(name="context_to_mu_W",
                                              shape=[100,
                                                     FLAGS.LATENT_VARIABLE_SIZE],
                                              dtype=tf.float32,
                                              initializer=tf.random_normal_initializer(stddev=0.1))

            context_to_mu_b = tf.get_variable(name="context_to_mu_b",
                                              shape=[FLAGS.LATENT_VARIABLE_SIZE],
                                              dtype=tf.float32)

            context_to_logvar_W = tf.get_variable(
                                              name="context_to_logvar_W",
                                              shape=[100,
                                                    FLAGS.LATENT_VARIABLE_SIZE],
                                              dtype=tf.float32,
                                              initializer=tf.random_normal_initializer(stddev=0.1))

            context_to_logvar_b = tf.get_variable(
                                              name="context_to_logvar_b",
                                              shape=[FLAGS.LATENT_VARIABLE_SIZE],
                                              dtype=tf.float32)

        with tf.name_scope("rnn_output_and_label"):
            rnn_output_and_label = tf.concat((encoder_rnn_output, self.label_onehot),
                                             axis=1,
                                             name="concat_encoder_rnn_output_and_label")

        with tf.name_scope("sampler_hiddenstate"):
            h = tf.nn.relu(tf.matmul(rnn_output_and_label, context_to_hidden_W) + context_to_hidden_b)

        with tf.name_scope("mu"):
            self.mu = tf.matmul(h, context_to_mu_W) + context_to_mu_b
        with tf.name_scope("log_var"):
            self.logvar = tf.matmul(h, context_to_logvar_W) + context_to_logvar_b

        with tf.name_scope("z"):
            z = tf.truncated_normal((FLAGS.BATCH_SIZE, FLAGS.LATENT_VARIABLE_SIZE), stddev=1.0)

        with tf.name_scope("latent_variables"):
            self.latent_variables = self.mu + tf.exp(0.5 * self.logvar) * z


class Encoder_vae(object):
    def __init__(self, embedding, encoder_input_list, is_training=True, ru=False):
        with tf.variable_scope("Encoder_input"):
            self.embedding = embedding
            self.encoder_input_list = encoder_input_list

            self.is_training = is_training

        with tf.variable_scope("encoder_rnn"):
            with tf.variable_scope("rnn_input_weight"):
                self.rnn_input_W = tf.get_variable(name="rnn_input_W",
                                                   shape=(FLAGS.EMBED_SIZE, FLAGS.RNN_SIZE),
                                                   dtype=tf.float32,
                                                   initializer=tf.random_normal_initializer(stddev=0.1))
                self.rnn_input_b = tf.get_variable(name="rnn_input_b",
                                                   shape=(FLAGS.RNN_SIZE),
                                                   dtype=tf.float32)

            with tf.variable_scope("encoder_rnn"):
                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(FLAGS.RNN_SIZE)

                if self.is_training:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                         output_keep_prob=FLAGS.ENCODER_DROPOUT_KEEP)

                self.cell = tf.contrib.rnn.MultiRNNCell([cell] * FLAGS.RNN_NUM)

                self.init_states = [cell.zero_state(FLAGS.BATCH_SIZE, tf.float32)
                                    for _ in range(FLAGS.RNN_NUM)]
                self.states = [tf.placeholder(tf.float32,
                                              (FLAGS.BATCH_SIZE),
                                              name="state")
                               for _ in range(FLAGS.RNN_NUM)]

            with tf.variable_scope("encoder_rnn_output"):
                self.encoder_rnn_output = self.rnn_train_predict()


        with tf.variable_scope("encoder_linear1"):
            context_to_hidden_W = tf.get_variable(name="context_to_hidden_W",
                                                  shape=[FLAGS.RNN_SIZE,
                                                         100],
                                                  dtype=tf.float32,
                                                  initializer=tf.random_normal_initializer(stddev=0.1))

            context_to_hidden_b = tf.get_variable(name="context_to_hidden_b",
                                                  shape=[100],
                                                  dtype=tf.float32)


        with tf.variable_scope("encoder_linear2"):
            context_to_mu_W = tf.get_variable(name="context_to_mu_W",
                                              shape=[100,
                                                     FLAGS.LATENT_VARIABLE_SIZE],
                                              dtype=tf.float32,
                                              initializer=tf.random_normal_initializer(stddev=0.1))

            context_to_mu_b = tf.get_variable(name="context_to_mu_b",
                                              shape=[FLAGS.LATENT_VARIABLE_SIZE],
                                              dtype=tf.float32)

            context_to_logvar_W = tf.get_variable(
                                              name="context_to_logvar_W",
                                              shape=[100,
                                                    FLAGS.LATENT_VARIABLE_SIZE],
                                              dtype=tf.float32,
                                              initializer=tf.random_normal_initializer(stddev=0.1))

            context_to_logvar_b = tf.get_variable(
                                              name="context_to_logvar_b",
                                              shape=[FLAGS.LATENT_VARIABLE_SIZE],
                                              dtype=tf.float32)

        with tf.name_scope("hiddenstate"):
            h = tf.nn.relu(tf.matmul(self.encoder_rnn_output, context_to_hidden_W) + context_to_hidden_b)

        with tf.name_scope("mu"):
            self.mu = tf.matmul(h, context_to_mu_W) + context_to_mu_b
        with tf.name_scope("log_var"):
            self.logvar = tf.matmul(h, context_to_logvar_W) + context_to_logvar_b

        with tf.name_scope("z"):
            z = tf.truncated_normal((FLAGS.BATCH_SIZE, FLAGS.LATENT_VARIABLE_SIZE), stddev=1.0)

        with tf.name_scope("latent_variables"):
            self.latent_variables = self.mu + tf.exp(0.5 * self.logvar) * z


    # input text from dataset
    def rnn_train_predict(self):
        pred = []
        state = self.init_states
        for i in range(FLAGS.SEQ_LEN):
            with tf.name_scope("encoder_input_embedding"):
                encoder_input = self.encoder_input_list[i]
                encoder_input_embedding = tf.nn.embedding_lookup(self.embedding, encoder_input)
                assert encoder_input_embedding.shape == (FLAGS.BATCH_SIZE,
                                                         FLAGS.EMBED_SIZE)

            with tf.name_scope("rnn_input"):
                rnn_input = tf.nn.relu(tf.matmul(encoder_input_embedding, self.rnn_input_W) + self.rnn_input_b)
                assert rnn_input.shape == (FLAGS.BATCH_SIZE, FLAGS.RNN_SIZE)

            with tf.name_scope("rnn_predict"):
                step_pred, state = self.cell(rnn_input, state)
                assert state[-1][1].shape == (FLAGS.BATCH_SIZE, FLAGS.RNN_SIZE)
                assert step_pred.shape == (FLAGS.BATCH_SIZE, FLAGS.RNN_SIZE)

                pred.append(step_pred)

        return state[-1][1] # last hidden state


Encoder = {
    "Encoder_vae": Encoder_vae,
    "Encoder_cvae" : Encoder_cvae
}
