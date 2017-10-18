import numpy as np
import tensorflow as tf

from train_vae import log_and_print
from model.vae import VAE
from config import FLAGS
from utils.batchloader import BatchLoader

LOG_DIR = "log/DIRNAME"
MODEL_DIR = LOG_DIR + "/model"
SAVE_FILE = LOG_DIR + "/sampled_texts.txt"
SAMPLE_NUM = 128

def sampling():
    batchloader = BatchLoader(with_label=True)

    sess_conf = tf.ConfigProto(
        gpu_options = tf.GPUOptions(
            # allow_growth = True
        )
    )

    with tf.Graph().as_default():
        with tf.Session(config=sess_conf) as sess:
            with tf.variable_scope("VAE"):
                vae_restored = VAE[FLAGS.VAE_NAME](batchloader, is_training=False, ru=False)

            saver = tf.train.Saver()
            saver.restore(sess, MODEL_DIR + "/model40.ckpt")

            itr = SAMPLE_NUM // FLAGS.BATCH_SIZE
            res = SAMPLE_NUM - itr * FLAGS.BATCH_SIZE

            generated_texts = []
            lv_list= []
            for i in range(itr+1):
                z = np.random.normal(loc=0.0, scale=1.0,
                                     size=[FLAGS.BATCH_SIZE, FLAGS.LATENT_VARIABLE_SIZE])
                sample_logits = sess.run(vae_restored.logits,
                                         feed_dict = {vae_restored.latent_variables: z})
                lv_list.extend(z)

                if i==itr:
                    sample_num = res
                else:
                    sample_num = FLAGS.BATCH_SIZE

                sample_texts = batchloader.logits2str(logits=sample_logits, sample_num=sample_num)
                generated_texts.extend(sample_texts)

            for i in range(SAMPLE_NUM):
                log_and_print(SAVE_FILE, generated_texts[i])

if __name__ == "__main__":
    sampling()
