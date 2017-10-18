import sys

import tensorflow as tf

sys.path.append("../")

from config import FLAGS


class Decoder_cvae(object):
    def __init__(self, decoder_input, latent_variables, label_onehot,
                 embedding, batchloader, is_training=True, ru=False):
        with tf.name_scope("decoder_input"):
            self.decoder_input = decoder_input
            self.latent_variables = latent_variables
            self.embedding = embedding
            self.label_onehot = label_onehot

            self.batchloader = batchloader
            self.go_input = tf.constant(self.batchloader.go_input,
                                        dtype=tf.int32)
            self.is_training = is_training

        with tf.variable_scope("lv2decoder"):
            decoder_input_embedding = tf.nn.embedding_lookup(self.embedding, decoder_input)

            assert decoder_input_embedding.shape == (FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN, FLAGS.EMBED_SIZE)

            with tf.variable_scope("decoder_lv_linear"):
                self.decoder_lv_W = tf.get_variable(name="decoder_lv_W",
                                                   shape=(FLAGS.LATENT_VARIABLE_SIZE + FLAGS.LABEL_CLASS,
                                                          FLAGS.DECODER_CNN_EXTERNAL_CHANNEL
                                                          - FLAGS.EMBED_SIZE),
                                                   dtype=tf.float32,
                                                   initializer=tf.random_normal_initializer(stddev=0.1))
                self.decoder_lv_b = tf.get_variable(name="decoder_lv_b",
                                                   shape=(FLAGS.DECODER_CNN_EXTERNAL_CHANNEL
                                                          - FLAGS.EMBED_SIZE),
                                                   dtype=tf.float32)

            with tf.name_scope("lv_and_label"):
                lv_and_label = tf.concat((self.latent_variables, self.label_onehot),
                                         axis=1,
                                         name="concat_decoder_lv_and_label")

            with tf.name_scope("lv_reshape"):
                latent_variables_reshaped = tf.matmul(lv_and_label, self.decoder_lv_W) + self.decoder_lv_b

                if FLAGS.DECODER_BATCHNORM:
                    latent_variables_reshaped = tf.contrib.layers.batch_norm(
                                                    latent_variables_reshaped,
                                                    decay=0.99,
                                                    center=True,
                                                    scale=True,
                                                    updates_collections=None,
                                                    is_training = self.is_training,
                                                    scope="lv_reshape",
                                                    fused=True)

                latent_variables_reshaped = tf.nn.relu(latent_variables_reshaped)

                if self.is_training:
                    latent_variables_reshaped = tf.nn.dropout(latent_variables_reshaped,
                                                              keep_prob=FLAGS.DECODER_DROPOUT_KEEP)

            with tf.name_scope("lv_matrix"):
                latent_variables_matrix = tf.convert_to_tensor(
                                                [latent_variables_reshaped for _ in range(FLAGS.SEQ_LEN)])
                latent_variables_matrix = tf.transpose(latent_variables_matrix, perm=[1, 0, 2])

            with tf.name_scope("decoder_cnn_input"):
                decoder_cnn_input = tf.concat((decoder_input_embedding, latent_variables_matrix),
                                              axis=2,
                                              name="concat_decoder_input_and_lv")
                assert decoder_cnn_input.shape == (FLAGS.BATCH_SIZE,
                                                   FLAGS.SEQ_LEN,
                                                   FLAGS.DECODER_CNN_EXTERNAL_CHANNEL)

        with tf.variable_scope("decoder_cnn"):
            with tf.variable_scope("decoder_cnn2vocab"):
                self.cnn2vocab_W = tf.get_variable(name="decoder_cnn2vocab_W",
                                                   shape=(FLAGS.DECODER_CNN_EXTERNAL_CHANNEL,
                                                          FLAGS.VOCAB_SIZE),
                                                   dtype=tf.float32,
                                                   initializer=tf.random_normal_initializer(stddev=0.1))

                self.cnn2vocab_b = tf.get_variable(name="decoder_cnn2vocab_b",
                                                   shape=(FLAGS.VOCAB_SIZE),
                                                   dtype=tf.float32)

            if self.is_training:
                decoder_cnn_output = self.cnn_unit(decoder_cnn_input, is_training=True)
            else:
                decoder_cnn_output = self.test_cnn(latent_variables_matrix)

            with tf.name_scope("decoder_cnn_output"):
                decoder_cnn_output_t = tf.transpose(decoder_cnn_output,
                                                    perm=[1, 0, 2],
                                                    name="transpose_decoder_cnn_output")

                self.decoder_cnn_output_list = []
                for i in range(FLAGS.SEQ_LEN):
                    self.decoder_cnn_output_list.append(decoder_cnn_output_t[i])
                    assert self.decoder_cnn_output_list[i].shape == (FLAGS.BATCH_SIZE, FLAGS.DECODER_CNN_EXTERNAL_CHANNEL)

            with tf.name_scope("logits"):
                self.logits = []
                for cnn_output in self.decoder_cnn_output_list:
                    logit = tf.matmul(cnn_output, self.cnn2vocab_W) + self.cnn2vocab_b
                    assert logit.shape == (FLAGS.BATCH_SIZE, FLAGS.VOCAB_SIZE)

                    self.logits.append(logit)


    def cnn_unit(self, decoder_cnn_input, is_training=True, reuse=False):
        next_input = tf.transpose(decoder_cnn_input,
                                  perm=[0, 2, 1],
                                  name="transpose_cnn_input_to_NCW")
        next_input = tf.reshape(next_input,
                                shape=(FLAGS.BATCH_SIZE,
                                       FLAGS.DECODER_CNN_EXTERNAL_CHANNEL,
                                       1,
                                       FLAGS.SEQ_LEN))

        for i in range(FLAGS.DECODER_CNN_LAYER_NUM):
            layer_name = "decoder_cnn%d" % (i+1)
            with tf.variable_scope(layer_name, reuse=reuse):

                # ResNet
                res = next_input

                with tf.name_scope("h1"):
                    filter1 = tf.get_variable(name=layer_name + "_filter1",
                                             shape=(1,
                                                    1,
                                                    FLAGS.DECODER_CNN_EXTERNAL_CHANNEL,
                                                    FLAGS.DECODER_CNN_INTERNAL_CHANNEL),
                                             dtype=tf.float32,
                                             initializer=tf.random_normal_initializer(stddev=0.1))

                    h1 = tf.nn.convolution(next_input,
                                           filter1,
                                           padding='VALID',
                                           strides=[1, 1],
                                           dilation_rate=None,
                                           name=layer_name + "_conv1d_1",
                                           data_format='NCHW')
                    assert h1.shape == (FLAGS.BATCH_SIZE, FLAGS.DECODER_CNN_INTERNAL_CHANNEL, 1, FLAGS.SEQ_LEN)

                    if FLAGS.DECODER_BATCHNORM:
                        h1 = tf.contrib.layers.batch_norm(
                                                        h1,
                                                        decay=0.99,
                                                        center=True,
                                                        scale=True,
                                                        updates_collections=None,
                                                        is_training = self.is_training,
                                                        scope=layer_name + '_bn1',
                                                        fused=True)

                    h1 = tf.nn.relu(h1)

                    if self.is_training:
                        h1 = tf.nn.dropout(h1, keep_prob=FLAGS.DECODER_DROPOUT_KEEP)

                    pad =  tf.zeros([FLAGS.BATCH_SIZE,
                                     FLAGS.DECODER_CNN_INTERNAL_CHANNEL,
                                     1,
                                     FLAGS.DECODER_CNN_PAD[i]])
                    h1 = tf.concat([pad, h1, pad], axis=3)

                with tf.name_scope("h2"):
                    filter2 = tf.get_variable(name=layer_name + "_filter2",
                                             shape=(1,
                                                    FLAGS.DECODER_CNN_FILTER_SIZE,
                                                    FLAGS.DECODER_CNN_INTERNAL_CHANNEL,
                                                    FLAGS.DECODER_CNN_INTERNAL_CHANNEL),
                                             dtype=tf.float32,
                                             initializer=tf.random_normal_initializer(stddev=0.1))

                    h2 = tf.nn.convolution(h1,
                                           filter2,
                                           padding='VALID',
                                           strides=[1, 1],
                                           dilation_rate=[1, FLAGS.DECODER_CNN_DILATION[i]],
                                           name=layer_name + "_conv1d_2",
                                           data_format='NCHW')
                    assert h2.shape[3] - FLAGS.DECODER_CNN_PAD[i] == FLAGS.SEQ_LEN

                    h2 = h2[:, :, :, :FLAGS.SEQ_LEN]
                    assert h2.shape == (FLAGS.BATCH_SIZE, FLAGS.DECODER_CNN_INTERNAL_CHANNEL, 1, FLAGS.SEQ_LEN)

                    if FLAGS.DECODER_BATCHNORM:
                        h2 = tf.contrib.layers.batch_norm(
                                                        h2,
                                                        decay=0.99,
                                                        center=True,
                                                        scale=True,
                                                        updates_collections=None,
                                                        is_training = self.is_training,
                                                        scope=layer_name + '_bn2',
                                                        fused=True)

                    h2 = tf.nn.relu(h2)

                    if self.is_training:
                        h2 = tf.nn.dropout(h2, keep_prob=FLAGS.DECODER_DROPOUT_KEEP)

                with tf.name_scope("h3"):
                    filter3 = tf.get_variable(name=layer_name + "_filter3",
                                             shape=(1,
                                                    1,
                                                    FLAGS.DECODER_CNN_INTERNAL_CHANNEL,
                                                    FLAGS.DECODER_CNN_EXTERNAL_CHANNEL),
                                             dtype=tf.float32,
                                             initializer=tf.random_normal_initializer(stddev=0.1))

                    h3 = tf.nn.convolution(h2,
                                           filter3,
                                           padding='VALID',
                                           strides=[1, 1],
                                           dilation_rate=None,
                                           name=layer_name + "_conv1d_3",
                                           data_format='NCHW')
                    assert h3.shape == (FLAGS.BATCH_SIZE, FLAGS.DECODER_CNN_EXTERNAL_CHANNEL, 1, FLAGS.SEQ_LEN)

                    if FLAGS.DECODER_BATCHNORM:
                        h3 = tf.contrib.layers.batch_norm(
                                                        h3,
                                                        decay=0.99,
                                                        center=True,
                                                        scale=True,
                                                        updates_collections=None,
                                                        is_training = self.is_training,
                                                        scope=layer_name + '_bn3',
                                                        fused=True)

                with tf.name_scope("next_input"):
                    next_input = h3

                    # ResNet
                    next_input = tf.nn.relu(next_input + res)

        with tf.name_scope("output"):
            next_input = tf.reshape(next_input,
                                    shape=(FLAGS.BATCH_SIZE,
                                           FLAGS.DECODER_CNN_EXTERNAL_CHANNEL,
                                           FLAGS.SEQ_LEN))
            next_input = tf.transpose(next_input,
                                      perm=[0, 2, 1],
                                      name="transpose_cnn_output_to_NWC")

        return next_input # last output


    def test_cnn(self, latent_variables_matrix):
        go_embeddings = tf.nn.embedding_lookup(self.embedding, self.go_input)
        go_embeddings = tf.reshape(go_embeddings,
                                   shape=(FLAGS.BATCH_SIZE, 1, FLAGS.EMBED_SIZE))

        zeros = tf.zeros([FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN-1, FLAGS.EMBED_SIZE])
        decoder_input_embedding = tf.concat((go_embeddings, zeros),
                                            axis=1,
                                            name="concat_go_and_zeros")

        next_input = tf.concat((decoder_input_embedding, latent_variables_matrix),
                                axis=2,
                                name="concat_decoder_input_and_lv_test_0")
        assert next_input.shape == (FLAGS.BATCH_SIZE,
                                    FLAGS.SEQ_LEN,
                                    FLAGS.DECODER_CNN_EXTERNAL_CHANNEL)

        reuse = tf.get_variable_scope().reuse
        for i in range(FLAGS.SEQ_LEN):
            cnn_output = self.cnn_unit(next_input, is_training=False, reuse=reuse)
            reuse = True # whenever reuse should be True from second iteration

            cnn_output_t = tf.transpose(cnn_output, perm=[1, 0, 2])
            cnn_output_list = []
            for i in range(FLAGS.SEQ_LEN):
                cnn_output_list.append(cnn_output_t[i])
                assert cnn_output_list[i].shape == (FLAGS.BATCH_SIZE, FLAGS.DECODER_CNN_EXTERNAL_CHANNEL)

            logits = []
            for cnn_one_output in cnn_output_list:
                logit = tf.matmul(cnn_one_output, self.cnn2vocab_W) + self.cnn2vocab_b
                assert logit.shape == (FLAGS.BATCH_SIZE, FLAGS.VOCAB_SIZE)

                logits.append(logit)
            logits = tf.transpose(tf.convert_to_tensor(logits),
                                  perm=[1, 0, 2],
                                  name="transpose_logits_of_next_input_%d" % (i+1))
            output_symbol = tf.stop_gradient(tf.argmax(logits, 2))
            output_embedding = tf.nn.embedding_lookup(self.embedding, output_symbol)

            next_input_embedding = tf.concat((go_embeddings, output_embedding[:, :FLAGS.SEQ_LEN-1]),
                                             axis=1,
                                             name="concat_go_and_output_%d" % (i+1))
            next_input = tf.concat((next_input_embedding, latent_variables_matrix),
                                    axis=2,
                                    name="concat_decoder_input_and_lv_test_%d" % (i+1))

        return cnn_output


class Decoder_vae(object):
    def __init__(self, decoder_input, latent_variables, embedding,
                 batchloader, is_training=True, ru=False):
        with tf.name_scope("decoder_input"):
            self.decoder_input = decoder_input
            self.latent_variables = latent_variables
            self.embedding = embedding
            self.batchloader = batchloader
            self.go_input = tf.constant(self.batchloader.go_input,
                                        dtype=tf.int32)

            self.is_training = is_training

        with tf.variable_scope("lv2decoder"):
            decoder_input_embedding = tf.nn.embedding_lookup(self.embedding, decoder_input)

            # word dropout
            if self.is_training:
                decoder_input_embedding = tf.nn.dropout(decoder_input_embedding,
                                                        noise_shape=[FLAGS.BATCH_SIZE,
                                                                     FLAGS.SEQ_LEN,
                                                                     1],
                                                        keep_prob=FLAGS.DECODER_DROPOUT_KEEP) * FLAGS.DECODER_DROPOUT_KEEP

            assert decoder_input_embedding.shape == (FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN, FLAGS.EMBED_SIZE)


            with tf.variable_scope("decoder_lv_linear"):
                self.decoder_lv_W = tf.get_variable(name="decoder_lv_W",
                                                   shape=(FLAGS.LATENT_VARIABLE_SIZE,
                                                          FLAGS.DECODER_CNN_EXTERNAL_CHANNEL
                                                          - FLAGS.EMBED_SIZE),
                                                   dtype=tf.float32,
                                                   initializer=tf.random_normal_initializer(stddev=0.1))
                self.decoder_lv_b = tf.get_variable(name="decoder_lv_b",
                                                   shape=(FLAGS.DECODER_CNN_EXTERNAL_CHANNEL
                                                          - FLAGS.EMBED_SIZE),
                                                   dtype=tf.float32)

            with tf.name_scope("lv_reshape"):
                latent_variables_reshaped = tf.matmul(self.latent_variables, self.decoder_lv_W) + self.decoder_lv_b

                if FLAGS.DECODER_BATCHNORM:
                    latent_variables_reshaped = tf.contrib.layers.batch_norm(
                                                    latent_variables_reshaped,
                                                    decay=0.99,
                                                    center=True,
                                                    scale=True,
                                                    updates_collections=None,
                                                    is_training = self.is_training,
                                                    scope="lv_reshape",
                                                    fused=True)

                latent_variables_reshaped = tf.nn.relu(latent_variables_reshaped)

            with tf.name_scope("lv_matrix"):
                latent_variables_matrix = tf.convert_to_tensor(
                                                [latent_variables_reshaped for _ in range(FLAGS.SEQ_LEN)])
                latent_variables_matrix = tf.transpose(latent_variables_matrix, perm=[1, 0, 2])

            with tf.name_scope("decoder_cnn_input"):
                decoder_cnn_input = tf.concat((decoder_input_embedding, latent_variables_matrix),
                                              axis=2,
                                              name="concat_decoder_input_and_lv")
                assert decoder_cnn_input.shape == (FLAGS.BATCH_SIZE,
                                                   FLAGS.SEQ_LEN,
                                                   FLAGS.DECODER_CNN_EXTERNAL_CHANNEL)

        with tf.variable_scope("Decoder_cnn"):
            with tf.variable_scope("decoder_cnn2vocab"):
                self.cnn2vocab_W = tf.get_variable(name="decoder_cnn2vocab_W",
                                                   shape=(FLAGS.DECODER_CNN_EXTERNAL_CHANNEL,
                                                          FLAGS.VOCAB_SIZE),
                                                   dtype=tf.float32,
                                                   initializer=tf.random_normal_initializer(stddev=0.1))

                self.cnn2vocab_b = tf.get_variable(name="decoder_cnn2vocab_b",
                                                   shape=(FLAGS.VOCAB_SIZE),
                                                   dtype=tf.float32)

            if self.is_training:
                decoder_cnn_output = self.cnn_unit(decoder_cnn_input, is_training=True)
            else:
                decoder_cnn_output = self.test_cnn(latent_variables_matrix)

            with tf.name_scope("decoder_cnn_output"):
                decoder_cnn_output_t = tf.transpose(decoder_cnn_output,
                                                    perm=[1, 0, 2],
                                                    name="transpose_decoder_cnn_output")

                self.decoder_cnn_output_list = []
                for i in range(FLAGS.SEQ_LEN):
                    self.decoder_cnn_output_list.append(decoder_cnn_output_t[i])
                    assert self.decoder_cnn_output_list[i].shape == (FLAGS.BATCH_SIZE, FLAGS.DECODER_CNN_EXTERNAL_CHANNEL)

            with tf.name_scope("logits"):
                self.logits = []
                for cnn_output in self.decoder_cnn_output_list:
                    logit = tf.matmul(cnn_output, self.cnn2vocab_W) + self.cnn2vocab_b
                    assert logit.shape == (FLAGS.BATCH_SIZE, FLAGS.VOCAB_SIZE)

                    self.logits.append(logit)


    def cnn_unit(self, decoder_cnn_input, is_training=True, reuse=False):
        next_input = tf.transpose(decoder_cnn_input,
                                  perm=[0, 2, 1],
                                  name="transpose_cnn_input_to_NCW")
        next_input = tf.reshape(next_input,
                                shape=(FLAGS.BATCH_SIZE,
                                       FLAGS.DECODER_CNN_EXTERNAL_CHANNEL,
                                       1,
                                       FLAGS.SEQ_LEN))

        for i in range(FLAGS.DECODER_CNN_LAYER_NUM):
            layer_name = "decoder_cnn%d" % (i+1)
            with tf.variable_scope(layer_name, reuse=reuse):

                # ResNet
                res = next_input

                with tf.name_scope("h1"):
                    filter1 = tf.get_variable(name=layer_name + "_filter1",
                                             shape=(1,
                                                    1,
                                                    FLAGS.DECODER_CNN_EXTERNAL_CHANNEL,
                                                    FLAGS.DECODER_CNN_INTERNAL_CHANNEL),
                                             dtype=tf.float32,
                                             initializer=tf.random_normal_initializer(stddev=0.1))

                    h1 = tf.nn.convolution(next_input,
                                           filter1,
                                           padding='VALID',
                                           strides=[1, 1],
                                           dilation_rate=None,
                                           name=layer_name + "_conv1d_1",
                                           data_format='NCHW')
                    assert h1.shape == (FLAGS.BATCH_SIZE, FLAGS.DECODER_CNN_INTERNAL_CHANNEL, 1, FLAGS.SEQ_LEN)

                    if FLAGS.DECODER_BATCHNORM:
                        h1 = tf.contrib.layers.batch_norm(
                                                        h1,
                                                        decay=0.99,
                                                        center=True,
                                                        scale=True,
                                                        updates_collections=None,
                                                        is_training = self.is_training,
                                                        scope=layer_name + '_bn1',
                                                        fused=True)

                    h1 = tf.nn.relu(h1)

                    if self.is_training:
                        h1 = tf.nn.dropout(h1, keep_prob=FLAGS.DECODER_DROPOUT_KEEP)

                    pad =  tf.zeros([FLAGS.BATCH_SIZE,
                                     FLAGS.DECODER_CNN_INTERNAL_CHANNEL,
                                     1,
                                     FLAGS.DECODER_CNN_PAD[i]])
                    h1 = tf.concat([pad, h1, pad], axis=3)

                with tf.name_scope("h2"):
                    filter2 = tf.get_variable(name=layer_name + "_filter2",
                                             shape=(1,
                                                    FLAGS.DECODER_CNN_FILTER_SIZE,
                                                    FLAGS.DECODER_CNN_INTERNAL_CHANNEL,
                                                    FLAGS.DECODER_CNN_INTERNAL_CHANNEL),
                                             dtype=tf.float32,
                                             initializer=tf.random_normal_initializer(stddev=0.1))

                    h2 = tf.nn.convolution(h1,
                                           filter2,
                                           padding='VALID',
                                           strides=[1, 1],
                                           dilation_rate=[1, FLAGS.DECODER_CNN_DILATION[i]],
                                           name=layer_name + "_conv1d_2",
                                           data_format='NCHW')
                    assert h2.shape[3] - FLAGS.DECODER_CNN_PAD[i] == FLAGS.SEQ_LEN

                    h2 = h2[:, :, :, :FLAGS.SEQ_LEN]
                    assert h2.shape == (FLAGS.BATCH_SIZE, FLAGS.DECODER_CNN_INTERNAL_CHANNEL, 1, FLAGS.SEQ_LEN)

                    if FLAGS.DECODER_BATCHNORM:
                        h2 = tf.contrib.layers.batch_norm(
                                                        h2,
                                                        decay=0.99,
                                                        center=True,
                                                        scale=True,
                                                        updates_collections=None,
                                                        is_training = self.is_training,
                                                        scope=layer_name + '_bn2',
                                                        fused=True)

                    h2 = tf.nn.relu(h2)

                    if self.is_training:
                        h2 = tf.nn.dropout(h2, keep_prob=FLAGS.DECODER_DROPOUT_KEEP)

                with tf.name_scope("h3"):
                    filter3 = tf.get_variable(name=layer_name + "_filter3",
                                             shape=(1,
                                                    1,
                                                    FLAGS.DECODER_CNN_INTERNAL_CHANNEL,
                                                    FLAGS.DECODER_CNN_EXTERNAL_CHANNEL),
                                             dtype=tf.float32,
                                             initializer=tf.random_normal_initializer(stddev=0.1))

                    h3 = tf.nn.convolution(h2,
                                           filter3,
                                           padding='VALID',
                                           strides=[1, 1],
                                           dilation_rate=None,
                                           name=layer_name + "_conv1d_3",
                                           data_format='NCHW')
                    assert h3.shape == (FLAGS.BATCH_SIZE, FLAGS.DECODER_CNN_EXTERNAL_CHANNEL, 1, FLAGS.SEQ_LEN)

                    if FLAGS.DECODER_BATCHNORM:
                        h3 = tf.contrib.layers.batch_norm(
                                                        h3,
                                                        decay=0.99,
                                                        center=True,
                                                        scale=True,
                                                        updates_collections=None,
                                                        is_training = self.is_training,
                                                        scope=layer_name + '_bn3',
                                                        fused=True)

                with tf.name_scope("next_input"):
                    next_input = h3

                    # ResNet
                    next_input = tf.nn.relu(next_input + res)

        with tf.name_scope("output"):
            next_input = tf.reshape(next_input,
                                    shape=(FLAGS.BATCH_SIZE,
                                           FLAGS.DECODER_CNN_EXTERNAL_CHANNEL,
                                           FLAGS.SEQ_LEN))
            next_input = tf.transpose(next_input,
                                      perm=[0, 2, 1],
                                      name="transpose_cnn_output_to_NWC")

        return next_input # last output


    def test_cnn(self, latent_variables_matrix):
        go_embeddings = tf.nn.embedding_lookup(self.embedding, self.go_input)
        go_embeddings = tf.reshape(go_embeddings,
                                   shape=(FLAGS.BATCH_SIZE, 1, FLAGS.EMBED_SIZE))

        zeros = tf.zeros([FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN-1, FLAGS.EMBED_SIZE])
        decoder_input_embedding = tf.concat((go_embeddings, zeros),
                                            axis=1,
                                            name="concat_go_and_zeros")

        next_input = tf.concat((decoder_input_embedding, latent_variables_matrix),
                                axis=2,
                                name="concat_decoder_input_and_lv_test_0")
        assert next_input.shape == (FLAGS.BATCH_SIZE,
                                    FLAGS.SEQ_LEN,
                                    FLAGS.DECODER_CNN_EXTERNAL_CHANNEL)

        reuse = tf.get_variable_scope().reuse
        for i in range(FLAGS.SEQ_LEN):
            cnn_output = self.cnn_unit(next_input, is_training=False, reuse=reuse)
            reuse = True # whenever reuse should be True from second iteration

            cnn_output_t = tf.transpose(cnn_output, perm=[1, 0, 2])
            cnn_output_list = []
            for i in range(FLAGS.SEQ_LEN):
                cnn_output_list.append(cnn_output_t[i])
                assert cnn_output_list[i].shape == (FLAGS.BATCH_SIZE, FLAGS.DECODER_CNN_EXTERNAL_CHANNEL)

            logits = []
            for cnn_one_output in cnn_output_list:
                logit = tf.matmul(cnn_one_output, self.cnn2vocab_W) + self.cnn2vocab_b
                assert logit.shape == (FLAGS.BATCH_SIZE, FLAGS.VOCAB_SIZE)

                logits.append(logit)
            logits = tf.transpose(tf.convert_to_tensor(logits),
                                  perm=[1, 0, 2],
                                  name="transpose_logits_of_next_input_%d" % (i+1))
            output_symbol = tf.stop_gradient(tf.argmax(logits, 2))
            output_embedding = tf.nn.embedding_lookup(self.embedding, output_symbol)

            next_input_embedding = tf.concat((go_embeddings, output_embedding[:, :FLAGS.SEQ_LEN-1]),
                                             axis=1,
                                             name="concat_go_and_output_%d" % (i+1))
            next_input = tf.concat((next_input_embedding, latent_variables_matrix),
                                    axis=2,
                                    name="concat_decoder_input_and_lv_test_%d" % (i+1))

        return cnn_output


Decoder= {
    "Decoder_vae": Decoder_vae,
    "Decoder_cvae" : Decoder_cvae
}
