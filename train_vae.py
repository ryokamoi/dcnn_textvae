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
            batchloader = BatchLoader(with_label=False)

            with tf.variable_scope("VAE"):
                vae = VAE[FLAGS.VAE_NAME](batchloader, is_training=True, ru=False)

            with tf.variable_scope("VAE", reuse=True):
                vae_test = VAE[FLAGS.VAE_NAME](batchloader, is_training=False, ru=True)

            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(FLAGS.LOG_DIR, sess.graph)

            sess.run(tf.global_variables_initializer())

            log_and_print(log_file, "start training")

            loss_sum = []
            reconst_loss_sum = []
            kld_sum = []

            lr = FLAGS.LEARNING_RATE
            step = 0
            for epoch in range(FLAGS.EPOCH):
                log_and_print(log_file, "epoch %d" % (epoch+1))
                if epoch >= FLAGS.LR_DECAY_START and epoch % 2 == 0:
                    lr *= 0.5
                for batch in range(FLAGS.BATCHES_PER_EPOCH):

                    step += 1

                    encoder_input, decoder_input, target = \
                                        batchloader.next_batch(FLAGS.BATCH_SIZE, "train")
                    feed_dict = {vae.encoder_input: encoder_input,
                                 vae.decoder_input: decoder_input,
                                 vae.target: target,
                                 vae.step: step,
                                 vae.lr: lr}

                    logits, loss, reconst_loss, kld, merged_summary, _ \
                        = sess.run([vae.logits, vae.loss, vae.reconst_loss,
                                    vae.kld, vae.merged_summary, vae.train_op],
                                   feed_dict = feed_dict)

                    reconst_loss_sum.append(reconst_loss)
                    kld_sum.append(kld)
                    loss_sum.append(loss)
                    summary_writer.add_summary(merged_summary, step)

                    if(batch % 100 == 99):
                        log_and_print(log_file, "epoch %d batch %d" % \
                                                ((epoch+1), (batch+1)), br=False)

                        ave_loss = np.average(loss_sum)
                        log_and_print(log_file, "\tloss: %f" % ave_loss, br=False)
                        ave_rnnloss = np.average(reconst_loss_sum)
                        log_and_print(log_file, "\treconst_loss: %f" % ave_rnnloss, br=False)
                        ave_kld = np.average(kld_sum)
                        log_and_print(log_file, "\tkld %f" % ave_kld, br=False)

                        loss_sum = []
                        reconst_loss_sum = []
                        kld_sum = []

                        # train input, output
                        # output input and logits
                        sample_train_input, sample_train_input_list \
                            = sess.run([vae.encoder_input, vae.encoder_input_list],
                                       feed_dict = feed_dict)
                        encoder_input_texts = batchloader.logits2str(sample_train_input_list,
                                                                     1,
                                                                     onehot=False,
                                                                     numpy=True)

                        log_and_print(log_file, "\ttrain input: %s" % encoder_input_texts[0])
                        sample_train_outputs = batchloader.logits2str(logits, 1)
                        log_and_print(log_file, "\ttrain output: %s" % sample_train_outputs[0])


                        # validation output
                        sample_input, _, sample_target = batchloader.next_batch(FLAGS.BATCH_SIZE, "test")
                        sample_input_list, sample_latent_variables = \
                            sess.run([vae_test.encoder_input_list, vae_test.encoder.latent_variables],
                                     feed_dict = {vae_test.encoder_input: sample_input})
                        sample_logits, valid_loss, merged_summary = \
                                sess.run([vae_test.logits, vae_test.reconst_loss, vae_test.merged_summary],
                                          feed_dict = {vae_test.target: sample_target,
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
