# TensorFlow implementation of "Improved Variational Autoencoders for Text Modeling using Dilated Convolutions"

paper:[https://arxiv.org/abs/1702.08139v2](https://arxiv.org/abs/1702.08139v2)

This is NOT an original implementation. There may be some minor differences from the original structure.

Results are reported in [my blog](https://sesenosannko.github.io/contents/text_g/dcnn)

## Prerequisites

 * Python 3.5
 * tensorflow-gpu==1.3.0
 * matplotlib==2.0.2
 * numpy==1.13.1
 * scikit-learn==0.19.0


## Preparation

Dataset is not contained. Please prepare your own dataset.

 * Sentence

Pickle file of Numpy array of word ids (shape=[batch_size, sentence_length]).

 * Dictionary

Pickle file of Python dictionary. It should contain "<EOS>", "<PAD>", "<GO>" as meta words.

```python
  dictionary = {word1: id1,
                word2: id2,
                ...}
```

## Usage
### Simple VAE
#### Train

1. modify config.py
2. run

```bash
  python train_vae.py
```

#### Get sample sentences

1. modify sampling.py
2. run

```bash
  python sampling.py
```

### Semisupervised Classification

1. modify config.py
2. run


```bash
  python train_cvae.py
```

## License

MIT

## Author

Ryo Kamoi
