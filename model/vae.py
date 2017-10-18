import sys

import tensorflow as tf

sys.path.append("../")

from config import FLAGS
from encoder import Encoder, Sampler
from decoder import Decoder
from discriminator import Discriminator


class Semi_VAE(object):
    def __init__(self, batchloader, is_training=True, without_label=False, ru=False):
        self.batchloader = batchloader
        self.ru = ru
        self.is_training = is_training
        self.without_label = without_label

        self.lr = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        self.gumbel_temperature = tf.placeholder(tf.float32, shape=(), name="gumbel_temperature")

        with tf.name_scope("Placeholders"):
            self.encoder_input = tf.placeholder(tf.int64,
                                                shape=(FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN),
                                                name="encoder_input")

            self.decoder_input = tf.placeholder(tf.int64,
                                                shape=(FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN),
                                                name="decoder_input")

            self.target = tf.placeholder(tf.int64,
                                         shape=(FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN),
                                         name="target")

            encoder_input_t = tf.transpose(self.encoder_input, perm=[1, 0])
            self.encoder_input_list = []
            decoder_input_t = tf.transpose(self.decoder_input, perm=[1, 0])
            self.decoder_input_list = []
            target_t = tf.transpose(self.target, perm=[1, 0])
            self.target_list = []

            self.step = tf.placeholder(tf.float32, shape=(), name="step")

            for i in range(FLAGS.SEQ_LEN):
                self.encoder_input_list.append(encoder_input_t[i])
                assert self.encoder_input_list[i].shape == (FLAGS.BATCH_SIZE)

                self.decoder_input_list.append(decoder_input_t[i])
                assert self.decoder_input_list[i].shape == (FLAGS.BATCH_SIZE)

                self.target_list.append(target_t[i])
                assert self.target_list[i].shape == (FLAGS.BATCH_SIZE)


            if not without_label:
                self.label = tf.placeholder(tf.int64,
                                            shape=(FLAGS.BATCH_SIZE),
                                            name="label")

                self.label_onehot = tf.one_hot(self.label, FLAGS.LABEL_CLASS, name="label_onehot")
                assert self.label_onehot.shape == (FLAGS.BATCH_SIZE, FLAGS.LABEL_CLASS)


        with tf.variable_scope("Embedding"):
            self.embedding = tf.get_variable(name="embedding",
                                             shape=[FLAGS.VOCAB_SIZE, FLAGS.EMBED_SIZE],
                                             dtype=tf.float32,
                                             initializer=tf.random_normal_initializer(stddev=0.1))

        with tf.variable_scope("Encoder"):
            self.encoder = Encoder[FLAGS.ENCODER_NAME](
                                   self.embedding,
                                   self.encoder_input_list,
                                   is_training = self.is_training,
                                   ru = self.ru)

        with tf.variable_scope("Discriminator"):
            self.discriminator = Discriminator(self.encoder.encoder_rnn_output,
                                               self.gumbel_temperature)

            if self.without_label:
                self.label_onehot = self.discriminator.discriminator_sampling_onehot
                assert self.label_onehot.shape == (FLAGS.BATCH_SIZE, FLAGS.LABEL_CLASS)

        with tf.name_scope("Latent_variables"):
            self.sampler = Sampler(self.encoder.encoder_rnn_output,
                                   self.label_onehot,
                                   is_training = self.is_training)

            if self.is_training:
                self.latent_variables = self.sampler.latent_variables
            else:
                self.latent_variables = tf.placeholder(tf.float32,
                                                       shape=(FLAGS.BATCH_SIZE,
                                                              FLAGS.LATENT_VARIABLE_SIZE),
                                                       name="latent_variables_input")

        with tf.variable_scope("Decoder"):
            self.decoder = Decoder[FLAGS.DECODER_NAME](
                                           self.decoder_input,
                                           self.latent_variables,
                                           self.label_onehot,
                                           self.embedding,
                                           self.batchloader,
                                           is_training = self.is_training,
                                           ru = self.ru)

        with tf.name_scope("Loss"):
            if not self.without_label:
                discriminator_correct = tf.equal(self.discriminator.discriminator_predict,
                                                 self.label)
                self.discriminator_accuracy = tf.reduce_mean(tf.cast(discriminator_correct,
                                                                     tf.float32))
                self.discriminator_loss = tf.reduce_mean(
                                            tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                logits=self.discriminator.discriminator_logits,
                                                labels=self.label,
                                                name="labeled_discriminator_cross_entropy")) * FLAGS.SEQ_LEN
            else:
                true_y = tf.fill([FLAGS.BATCH_SIZE, FLAGS.LABEL_CLASS],
                                 1/FLAGS.LABEL_CLASS,
                                 name="true_y_distribution")
                self.kld2 = tf.reduce_mean(
                                tf.reduce_sum(self.discriminator.discriminator_prob * \
                                                (tf.log(self.discriminator.discriminator_prob + 1e-6) - \
                                                 tf.log(true_y)),
                                              axis=1))

            self.logits = self.decoder.logits

            self.kld = tf.reduce_mean(-0.5 *
                                      tf.reduce_sum(self.sampler.logvar
                                                    - tf.square(self.sampler.mu)
                                                    - tf.exp(self.sampler.logvar)
                                                    + 1,
                                                    axis=1))
            self.kld_weight = tf.clip_by_value(FLAGS.INIT_KLD_WEIGHT +
                                               (1-FLAGS.INIT_KLD_WEIGHT) *
                                                    (self.step - FLAGS.KLD_ANNEAL_START) /
                                                    (FLAGS.KLD_ANNEAL_END - FLAGS.KLD_ANNEAL_START),
                                               0, 1)

            reconst_losses = [tf.nn.sparse_softmax_cross_entropy_with_logits( \
                                                    logits=logits, labels=targets) \
                                        for logits, targets in zip(self.logits, self.target_list)]
            self.reconst_loss = tf.reduce_mean(reconst_losses) * FLAGS.SEQ_LEN

            if not self.without_label:
                self.loss = self.reconst_loss + self.kld_weight * self.kld \
                                + tf.log(1/FLAGS.LABEL_CLASS) + self.discriminator_loss
            else:
                self.loss = self.reconst_loss + self.kld_weight * self.kld + self.kld2


        with tf.name_scope("Summary"):
            if self.is_training and not self.without_label:
                reconst_loss_summary = tf.summary.scalar("labeled_reconst_loss", self.reconst_loss, family="train_loss")
                kld_summary = tf.summary.scalar("labeled_kld", self.kld, family="kld")
                disc_loss_summary = tf.summary.scalar("labeled_disc_train_loss", self.discriminator_loss, family="disc_loss")
                disc_acc_summary = tf.summary.scalar("labeled_disc_train_acc", self.discriminator_accuracy, family="disc_acc")

                kld_weight_summary = tf.summary.scalar("kld_weight", self.kld_weight, family="parameters")
                mu_summary = tf.summary.histogram("labeled_mu", tf.reduce_mean(self.sampler.mu, 0))
                var_summary = tf.summary.histogram("labeled_var", tf.reduce_mean(tf.exp(self.sampler.logvar), 0))
                lr_summary = tf.summary.scalar("lr", self.lr, family="parameters")

                self.merged_summary = tf.summary.merge([reconst_loss_summary, kld_summary,
                                                        disc_loss_summary, disc_acc_summary,
                                                        kld_weight_summary, mu_summary, var_summary,
                                                        lr_summary])
            elif self.is_training and self.without_label:
                reconst_loss_summary = tf.summary.scalar("unlabeled_reconst_loss", self.reconst_loss, family="train_loss")
                kld_summary = tf.summary.scalar("unlabeled_kld", self.kld, family="kld")
                gumbel_summary = tf.summary.scalar("gumbel_temperature", self.gumbel_temperature, family="parameters")
                kld2_summary = tf.summary.scalar("unlabeled_kld2", self.kld2, family="kld")

                mu_summary = tf.summary.histogram("unlabeled_mu", tf.reduce_mean(self.sampler.mu, 0))
                var_summary = tf.summary.histogram("unlabeled_var", tf.reduce_mean(tf.exp(self.sampler.logvar), 0))

                self.merged_summary = tf.summary.merge([reconst_loss_summary, kld_summary, gumbel_summary,
                                                        kld2_summary, mu_summary, var_summary])
            else:
                valid_reconst_loss_summary = tf.summary.scalar("valid_reconst_loss", self.reconst_loss, family="valid_loss")
                disc_loss_summary = tf.summary.scalar("disc_valid_loss", self.discriminator_loss, family="disc_loss")
                disc_acc_summary = tf.summary.scalar("disc_valid_acc", self.discriminator_accuracy, family="disc_acc")

                self.merged_summary = tf.summary.merge([valid_reconst_loss_summary, disc_loss_summary, disc_acc_summary])

        if self.is_training:
            tvars = tf.trainable_variables()
            with tf.name_scope("Optimizer"):
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                                  FLAGS.MAX_GRAD)
                optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

                self.train_op = optimizer.apply_gradients(zip(grads, tvars))


class Simple_VAE(object):
    def __init__(self, batchloader, is_training=True, ru=False):
        self.batchloader = batchloader
        self.ru = ru
        self.is_training = is_training

        self.lr = tf.placeholder(tf.float32, shape=(), name="learning_rate")

        with tf.name_scope("Placeholders"):
            self.encoder_input = tf.placeholder(tf.int64,
                                                shape=(FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN),
                                                name="encoder_input")

            self.decoder_input = tf.placeholder(tf.int64,
                                                shape=(FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN),
                                                name="decoder_input")

            self.target = tf.placeholder(tf.int64,
                                         shape=(FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN),
                                         name="target")

            encoder_input_t = tf.transpose(self.encoder_input, perm=[1, 0])
            self.encoder_input_list = []
            decoder_input_t = tf.transpose(self.decoder_input, perm=[1, 0])
            self.decoder_input_list = []
            target_t = tf.transpose(self.target, perm=[1, 0])
            self.target_list = []

            self.step = tf.placeholder(tf.float32, shape=(), name="step")

            for i in range(FLAGS.SEQ_LEN):
                self.encoder_input_list.append(encoder_input_t[i])
                assert self.encoder_input_list[i].shape == (FLAGS.BATCH_SIZE)

                self.decoder_input_list.append(decoder_input_t[i])
                assert self.decoder_input_list[i].shape == (FLAGS.BATCH_SIZE)

                self.target_list.append(target_t[i])
                assert self.target_list[i].shape == (FLAGS.BATCH_SIZE)


        with tf.variable_scope("Embedding"):
            self.embedding = tf.get_variable(name="embedding",
                                             shape=[FLAGS.VOCAB_SIZE, FLAGS.EMBED_SIZE],
                                             dtype=tf.float32,
                                             initializer=tf.random_normal_initializer(stddev=0.1))

        with tf.variable_scope("Encoder"):
            self.encoder = Encoder[FLAGS.ENCODER_NAME](
                                   self.embedding,
                                   self.encoder_input_list,
                                   is_training = self.is_training,
                                   ru = self.ru)

        with tf.name_scope("Latent_variables"):
            if self.is_training:
                self.latent_variables = self.encoder.latent_variables
            else:
                self.latent_variables = tf.placeholder(tf.float32,
                                                       shape=(FLAGS.BATCH_SIZE,
                                                              FLAGS.LATENT_VARIABLE_SIZE),
                                                       name="latent_variables_input")

        with tf.variable_scope("Decoder"):
            self.decoder = Decoder[FLAGS.DECODER_NAME](
                                           self.decoder_input,
                                           self.latent_variables,
                                           self.embedding,
                                           self.batchloader,
                                           is_training = self.is_training,
                                           ru = self.ru)

        with tf.name_scope("Loss"):
            self.logits = self.decoder.logits

            self.kld = tf.reduce_mean(-0.5 *
                                      tf.reduce_sum(self.encoder.logvar
                                                    - tf.square(self.encoder.mu)
                                                    - tf.exp(self.encoder.logvar)
                                                    + 1,
                                                    axis=1))
            self.kld_weight = tf.clip_by_value(FLAGS.INIT_KLD_WEIGHT +
                                               (1-FLAGS.INIT_KLD_WEIGHT) *
                                                    (self.step - FLAGS.KLD_ANNEAL_START) /
                                                    (FLAGS.KLD_ANNEAL_END - FLAGS.KLD_ANNEAL_START),
                                               0, 1)

            reconst_losses = [tf.nn.sparse_softmax_cross_entropy_with_logits( \
                                                    logits=logits, labels=targets) \
                                        for logits, targets in zip(self.logits, self.target_list)]
            self.reconst_loss = tf.reduce_mean(reconst_losses) * FLAGS.SEQ_LEN

            self.loss = self.reconst_loss + self.kld_weight * self.kld

        with tf.name_scope("Summary"):
            if is_training:
                loss_summary = tf.summary.scalar("loss", self.loss, family="train_loss")
                reconst_loss_summary = tf.summary.scalar("reconst_loss", self.reconst_loss, family="train_loss")
                kld_summary = tf.summary.scalar("kld", self.kld, family="kld")
                kld_weight_summary = tf.summary.scalar("kld_weight", self.kld_weight, family="parameters")
                mu_summary = tf.summary.histogram("mu", tf.reduce_mean(self.encoder.mu, 0))
                var_summary = tf.summary.histogram("var", tf.reduce_mean(tf.exp(self.encoder.logvar), 0))
                lr_summary = tf.summary.scalar("lr", self.lr, family="parameters")

                self.merged_summary = tf.summary.merge([loss_summary, reconst_loss_summary, kld_summary,
                                                        kld_weight_summary, mu_summary, var_summary,
                                                        lr_summary])
            else:
                valid_reconst_loss_summary = tf.summary.scalar("valid_reconst_loss", self.reconst_loss, family="valid_loss")

                self.merged_summary = tf.summary.merge([valid_reconst_loss_summary])

        if(self.is_training):
            tvars = tf.trainable_variables()
            with tf.name_scope("Optimizer"):
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                                  FLAGS.MAX_GRAD)
                optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

                self.train_op = optimizer.apply_gradients(zip(grads, tvars))


VAE = {
    "Simple_VAE": Simple_VAE,
    "Semi_VAE" : Semi_VAE
}
