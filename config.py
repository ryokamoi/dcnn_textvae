from datetime import datetime

import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('DATA_PATH', "dataset/DATA_FILE_PATH", "")
flags.DEFINE_string('LABEL_PATH', "dataset/LABEL_FILE_PATH", "")
flags.DEFINE_string('DICT_PATH', "dictionary/DICT_FILE_PATH", "")

flags.DEFINE_integer('VOCAB_SIZE', 20000, '')
flags.DEFINE_integer('BATCH_SIZE', 32, '')
flags.DEFINE_integer('SEQ_LEN', 60, '')
flags.DEFINE_integer('LABELED_NUM', 500, '')
flags.DEFINE_integer('LABEL_CLASS', 2, '')
flags.DEFINE_integer('EPOCH', 40, '')
flags.DEFINE_integer('BATCHES_PER_EPOCH', 3000, '')

flags.DEFINE_string('VAE_NAME', 'Simple_VAE', '')
flags.DEFINE_string('ENCODER_NAME', 'Encoder_vae', '')
flags.DEFINE_string('DECODER_NAME', 'Decoder_vae', '')

flags.DEFINE_integer('ENCODER_DROPOUT_KEEP', 0.7, '')
flags.DEFINE_integer('DECODER_DROPOUT_KEEP', 0.9, '')
flags.DEFINE_integer('DECODER_DROPWORD_KEEP', 0.6, '')
flags.DEFINE_integer('LEARNING_RATE', 0.001, '')
flags.DEFINE_integer('LR_DECAY_START', 30, '')
flags.DEFINE_integer('MAX_GRAD', 5.0, '')

flags.DEFINE_integer('EMBED_SIZE', 512, '')
flags.DEFINE_integer('LATENT_VARIABLE_SIZE', 32, '')

flags.DEFINE_integer('RNN_NUM', 1, '')
flags.DEFINE_integer('RNN_SIZE', 1024, '')

flags.DEFINE_boolean('DECODER_BATCHNORM', True, '')
flags.DEFINE_integer('DECODER_CNN_INTERNAL_CHANNEL', 512, '')
flags.DEFINE_integer('DECODER_CNN_EXTERNAL_CHANNEL', 1024, '')
flags.DEFINE_integer('DECODER_CNN_FILTER_SIZE', 3, '')

decoder_cnn_dilation = [1, 2, 4]
flags.DEFINE_integer('DECODER_CNN_LAYER_NUM', len(decoder_cnn_dilation), '')
flags.DEFINE_integer('DECODER_CNN_DILATION', decoder_cnn_dilation, '')
flags.DEFINE_integer('DECODER_CNN_PAD', [2, 4, 8], '')

flags.DEFINE_integer('INIT_KLD_WEIGHT', 0.01, '')
flags.DEFINE_integer('KLD_ANNEAL_START', 0, '')
flags.DEFINE_integer('KLD_ANNEAL_END', 40 * 1000, '')

flags.DEFINE_string('LOG_DIR', "log/log" + datetime.now().strftime("%y%m%d-%H%M"), "")

FLAGS = flags.FLAGS
