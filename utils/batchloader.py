import sys
import pickle as pkl

sys.path.append("../")

import numpy as np
import tensorflow as tf

from config import FLAGS


class BatchLoader:
    def __init__(self, with_label=True):
        self.with_label = with_label

        self.go_token = '<GO>'
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'

        with open(FLAGS.DATA_PATH, "rb") as f:
            data = pkl.load(f)

        if self.with_label:
            with open(FLAGS.LABEL_PATH, "rb") as f:
                self.label = pkl.load(f)

        with open(FLAGS.DICT_PATH, "rb") as f:
            self.char_to_idx = pkl.load(f)

        self.idx_to_char = {}
        for char, idx in self.char_to_idx.items():
            self.idx_to_char[idx] = char

        self.idx_to_char[self.char_to_idx[self.pad_token]] = '_'
        self.idx_to_char[self.char_to_idx[self.unk_token]] = '??'

        indexes = np.array([i for i in range(len(data))], dtype=np.int32)
        indexes = np.random.permutation(indexes)
        data = np.array([np.copy(data[index]) for index in indexes])

        if self.with_label:
            self.label = np.array([np.copy(self.label[index]) for index in indexes])

        self.split = len(data) // 10

        if with_label:
            self.valid_data, self.labeled_data, self.unlabeled_data = \
                                data[:self.split], \
                                data[self.split:self.split+FLAGS.LABELED_NUM], \
                                data[self.split+FLAGS.LABELED_NUM:]

            self.valid_label, self.train_label = \
                self.label[:self.split], self.label[self.split:self.split+FLAGS.LABELED_NUM]
        else:
            self.valid_data, self.train_data = data[:self.split], data[self.split:]

        self.go_input = self.go_input()


    def next_batch(self, batch_size, target: str):
        if target == 'train':
            if self.with_label:
                indexes = np.array(np.random.randint(len(self.labeled_data), size=batch_size))
                encoder_input = [np.copy(self.labeled_data[idx]).tolist() for idx in indexes]
                labeled_list = self.wrap_tensor(encoder_input)

                label = np.array([self.train_label[idx] for idx in indexes])
                labeled_list.append(label)

                indexes = np.array(np.random.randint(len(self.unlabeled_data), size=batch_size))
                encoder_input = [np.copy(self.unlabeled_data[idx]).tolist() for idx in indexes]
                unlabeled_list = self.wrap_tensor(encoder_input)

                return labeled_list + unlabeled_list

            else:
                indexes = np.array(np.random.randint(len(self.train_data), size=batch_size))
                encoder_input = [np.copy(self.train_data[idx]).tolist() for idx in indexes]

                return self.wrap_tensor(encoder_input)

        else:
            indexes = np.array(np.random.randint(len(self.valid_data), size=batch_size))
            encoder_input = [np.copy(self.valid_data[idx]).tolist() for idx in indexes]

            if self.with_label:
                label = np.array([np.copy(self.valid_label[idx]).tolist() for idx in indexes])
                labeled_list = self.wrap_tensor(encoder_input)
                labeled_list.append(label)

                return labeled_list

            else:
                return self.wrap_tensor(encoder_input)


    def wrap_tensor(self, input):
        encoder_input = np.copy(input)
        decoder_input = np.array([np.hstack(([self.char_to_idx[self.go_token]], line[:len(line)-1])) for line in np.copy(input)])
        if FLAGS.DECODER_DROPWORD_KEEP < 1.0:
            r = np.random.rand(FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN)
            for i in range(FLAGS.BATCH_SIZE):
                for j in range(FLAGS.SEQ_LEN):
                    if(r[i][j] > FLAGS.DECODER_DROPWORD_KEEP and \
                                    decoder_input[i][j] not in [self.char_to_idx[self.go_token], self.char_to_idx[self.pad_token]]):
                        decoder_input[i][j] = self.char_to_idx[self.unk_token]
        decoder_target = np.copy(input)

        return [encoder_input, decoder_input, decoder_target]


    def go_input(self):
        go_input = np.array([self.char_to_idx[self.go_token] for _ in range(FLAGS.BATCH_SIZE)])

        return go_input


    def logits2str(self, logits, sample_num, onehot=True, numpy=False):
        """ convert logits into texts
            Args:
                logits: list of np.array (if onehot: [batch_size, vocab_size] else: [batch_size])
            Output:
                list of texts (batch_size) """

        assert sample_num <= FLAGS.BATCH_SIZE
        generated_texts = []

        if onehot:
            indices = [np.argmax(l, 1) for l in logits]
        else:
            indices = logits

        seq_len = len(indices)
        assert seq_len == FLAGS.SEQ_LEN

        for i in range(sample_num):
            x = ''
            for j in range(seq_len):
                x += self.idx_to_char[indices[j][i]]
            generated_texts.append(x)

        return generated_texts
