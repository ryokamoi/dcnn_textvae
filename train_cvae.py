import os
import shutil

import numpy as np
import tensorflow as tf

from model.vae import VAE
from config import FLAGS
from utils.batchloader import BatchLoader

def log_and_print(log_file, logstr, br=True):
    print(logstr)

    if(br):
        logstr = logstr + "\n"
    with open(log_file, 'a') as f:
        f.write(logstr)

def main():
    os.mkdir(FLAGS.LOG_DIR)
    os.mkdir(FLAGS.LOG_DIR + "/model")
    log_file = FLAGS.LOG_DIR + "/log.txt"
    shutil.copyfile("config.py", FLAGS.LOG_DIR + "/config.py")
    shutil.copyfile("README.md", FLAGS.LOG_DIR + "/README.md")

    sess_conf = tf.ConfigProto(
        gpu_options = tf.GPUOptions(
            # allow_growth = True
        )
    )

    with tf.Graph().as_default():
        with tf.Session(config=sess_conf) as sess:
            batchloader = BatchLoader(with_label=True)

            with tf.variable_scope("VAE"):
                vae_labeled = VAE[FLAGS.VAE_NAME](batchloader,
                                                  is_training=True,
                                                  without_label=False,
                                                  ru=False)

            with tf.variable_scope("VAE", reuse=True):
                vae_unlabeled = VAE[FLAGS.VAE_NAME](batchloader,
                                                    is_training=True,
                                                    without_label=True,
                                                    ru=True)

            with tf.variable_scope("VAE", reuse=True):
                vae_test = VAE[FLAGS.VAE_NAME](batchloader,
                                               is_training=False,
                                               without_label=False,
                                               ru=True)

            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(FLAGS.LOG_DIR, sess.graph)

            sess.run(tf.global_variables_initializer())

            log_and_print(log_file, "start training")

            loss_sum = []
            labeled_kld_sum = []
            labeled_reconst_loss_sum = []
            discriminator_loss_sum = []
            unlabeled_kld_sum = []
            unlabeled_reconst_loss_sum = []

            lr = FLAGS.LEARNING_RATE
            step = 0
            gumbel_temperature = 1.0
            for epoch in range(FLAGS.EPOCH):
                log_and_print(log_file, "epoch %d" % (epoch+1))
                if epoch >= FLAGS.LR_DECAY_START and epoch % 2 == 0:
                    lr *= 0.5
                for batch in range(FLAGS.BATCHES_PER_EPOCH):

                    step += 1

                    if step % 100 == 99:
                        gumbel_temperature = max(0.5, np.exp(-0.00001 * step))

                    labeled_encoder_input, labeled_decoder_input, labeled_target, label, \
                        unlabeled_encoder_input, unlabeled_decoder_input, unlabeled_target = \
                                batchloader.next_batch(FLAGS.BATCH_SIZE, "train")

                    # labeled dataset
                    labeled_feed_dict = {vae_labeled.encoder_input: labeled_encoder_input,
                                         vae_labeled.decoder_input: labeled_decoder_input,
                                         vae_labeled.target: labeled_target,
                                         vae_labeled.label: label,
                                         vae_labeled.step: step,
                                         vae_labeled.lr: lr}

                    labeled_logits, labeled_loss, labeled_reconst_loss, labeled_kld, \
                        discriminator_loss, discriminator_accuracy, labeled_merged_summary, _ \
                            = sess.run([vae_labeled.logits, vae_labeled.loss, vae_labeled.reconst_loss,
                                        vae_labeled.kld, vae_labeled.discriminator_loss, \
                                        vae_labeled.discriminator_accuracy, vae_labeled.merged_summary, \
                                        vae_labeled.train_op],
                                        feed_dict = labeled_feed_dict)

                    labeled_reconst_loss_sum.append(labeled_reconst_loss)
                    labeled_kld_sum.append(labeled_kld)
                    discriminator_loss_sum.append(discriminator_loss)

                    summary_writer.add_summary(labeled_merged_summary, step)


                    # unlabeled dataset
                    unlabeled_feed_dict = {vae_unlabeled.encoder_input: unlabeled_encoder_input,
                                           vae_unlabeled.decoder_input: unlabeled_decoder_input,
                                           vae_unlabeled.target: unlabeled_target,
                                           vae_unlabeled.step: step,
                                           vae_unlabeled.lr: lr,
                                           vae_unlabeled.gumbel_temperature: gumbel_temperature}

                    unlabeled_logits, unlabeled_loss, unlabeled_reconst_loss, unlabeled_kld, \
                        unlabeled_merged_summary, _ \
                            = sess.run([vae_unlabeled.logits, vae_unlabeled.loss, vae_unlabeled.reconst_loss,
                                        vae_unlabeled.kld, vae_unlabeled.merged_summary, vae_unlabeled.train_op],
                                        feed_dict = unlabeled_feed_dict)

                    unlabeled_reconst_loss_sum.append(unlabeled_reconst_loss)
                    unlabeled_kld_sum.append(unlabeled_kld)
                    loss_sum.append(labeled_loss + unlabeled_loss)

                    summary_writer.add_summary(unlabeled_merged_summary, step)


                    # log
                    if(batch == 9 or batch % 100 == 99):
                        log_and_print(log_file, "epoch %d batch %d" % \
                                                ((epoch+1), (batch+1)), br=False)

                        ave_loss = np.average(loss_sum)
                        log_and_print(log_file, "\tloss: %f" % ave_loss, br=False)
                        labeled_ave_rnnloss = np.average(labeled_reconst_loss_sum)
                        log_and_print(log_file, "\tlabeled_reconst_loss: %f" % labeled_ave_rnnloss, br=False)
                        labeled_ave_kld = np.average(labeled_kld_sum)
                        log_and_print(log_file, "\tlabeled_kld %f" % labeled_ave_kld, br=True)

                        unlabeled_ave_rnnloss = np.average(unlabeled_reconst_loss_sum)
                        log_and_print(log_file, "\tunlabeled_reconst_loss: %f" % unlabeled_ave_rnnloss, br=False)
                        unlabeled_ave_kld = np.average(unlabeled_kld_sum)
                        log_and_print(log_file, "\tunlabeled_kld %f" % unlabeled_ave_kld, br=True)

                        ave_disc_loss = np.average(discriminator_loss_sum)
                        log_and_print(log_file, "\tdisc_loss %f" % ave_disc_loss, br=True)


                        loss_sum = []
                        labeled_kld_sum = []
                        labeled_reconst_loss_sum = []
                        discriminator_loss_sum = []
                        unlabeled_kld_sum = []
                        unlabeled_reconst_loss_sum = []


                        # train input, output
                        # output input and logits
                        sample_train_input, sample_train_input_list \
                            = sess.run([vae_labeled.encoder_input, vae_labeled.encoder_input_list],
                                       feed_dict = labeled_feed_dict)
                        encoder_input_texts = batchloader.logits2str(sample_train_input_list,
                                                                     1,
                                                                     onehot=False,
                                                                     numpy=True)

                        log_and_print(log_file, "\ttrain input: %s" % encoder_input_texts[0])
                        sample_train_outputs = batchloader.logits2str(labeled_logits, 1)
                        log_and_print(log_file, "\ttrain output: %s" % sample_train_outputs[0])


                        # debug
                        train_latent_variables = \
                                        sess.run(vae_test.sampler.latent_variables,
                                                 feed_dict = {vae_test.encoder_input: sample_train_input,
                                                              vae_test.label: label})
                        sample_logits = sess.run(vae_test.logits,
                                                 feed_dict = {vae_test.latent_variables: train_latent_variables,
                                                              vae_test.label: label})
                        train_valid_samples = batchloader.logits2str(sample_logits, 1)
                        print("\ttrain valid output: %s" % train_valid_samples[0])


                        # sample output
                        sample_input, _, sample_target, sample_label = batchloader.next_batch(FLAGS.BATCH_SIZE, "test")
                        sample_input_list, sample_latent_variables, discriminator_loss, discriminator_accuracy = \
                            sess.run([vae_test.encoder_input_list, vae_test.sampler.latent_variables,
                                      vae_test.discriminator_loss, vae_test.discriminator_accuracy],
                                      feed_dict = {vae_test.encoder_input: sample_input,
                                                   vae_test.label: sample_label})
                        sample_logits, valid_loss, merged_summary = \
                                sess.run([vae_test.logits, vae_test.reconst_loss, vae_test.merged_summary],
                                          feed_dict = {vae_test.encoder_input: sample_input,
                                                       vae_test.target: sample_target,
                                                       vae_test.label: sample_label,
                                                       vae_test.latent_variables: sample_latent_variables})

                        log_and_print(log_file, "\tvalid loss: %f" % valid_loss)
                        sample_input_texts = batchloader.logits2str(sample_input_list,
                                                                    1, onehot=False, numpy=True)
                        sample_output_texts = batchloader.logits2str(sample_logits, 1)
                        log_and_print(log_file, "\tsample input: %s" % sample_input_texts[0])
                        log_and_print(log_file, "\tsample output: %s" % sample_output_texts[0])

                        summary_writer.add_summary(merged_summary, step)


                # save model
                save_path = saver.save(sess, FLAGS.LOG_DIR + ("/model/model%d.ckpt" % (epoch+1)))
                log_and_print(log_file, "Model saved in file %s" % save_path)


if __name__ == "__main__":
    main()
