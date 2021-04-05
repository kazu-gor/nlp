import time

import MeCab
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize(sentence):
    """Divide a Japanese sentence into columns of morphemes

    Args:
        sentence (str): Japanese sentence
    Returns:
        list: tokenized sentence
    """
    node = tagger.parse(sentence)
    node = node.split("\n")
    tokenized_sentence = []
    for i in range(len(node)):
        feature = node[i].split("\t")
        if feature[0] == "EOS":
            break
        tokenized_sentence.append(feature[0])
    return tokenized_sentence

""" test code """
tagger = MeCab.Tagger('-Ochasen')
sentence = "坊主が屏風に上手に坊主の絵を描いた"
print(f"sentence:\n{sentence}")
print(f"tokenized_sentence:\n{tokenize(sentence)}")
""" test code """

# define token and ID
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
PAD = 0
UNK = 1

# initialize dictionary
word2id = {
    PAD_TOKEN: PAD,
    UNK_TOKEN: UNK,
}

# Minimum number of appearances
MIN_COUNT = 1

class Vocab(object):
    """ Vocabulary management class """
    def __init__(self, word2id={}):
        """
        Args:
            word2id (dict, optional): Defaults to {}.
        """
        self.word2id = dict(word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def build_vocab(self, sentences, min_count=1):
        """
        Args:
            sentences (list of list of str): corpus.
            min_count (int): Defaults to 1.
        """
        word_counter = {}
        for sentence in sentences:
            for word in sentence:
                # dict.get(word, 0): if word exists -> return word, else 0
                word_counter[word] = word_counter.get(word, 0) + 1

        for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):
            if count < min_count:
                break
            _id = len(self.word2id)
            self.word2id.setdefault(word, _id)
            self.id2word[_id] = word

        self.raw_vocab = {w: word_counter[w] for w in self.word2id.keys() if w in word_counter}

""" test code """
vocab = Vocab(word2id = word2id)
vocab.build_vocab(sentence, min_count=MIN_COUNT)
""" test code """

def sentence_to_ids(vocab, sen):
    """word list -> list of id

    Args:
        vocab (class): object
        sen (lust of str): sentence

    Returns:
        list of str: list of word ID
    """
    out = [vocab.word2id.get(word, UNK) for word in sen]
    return out

""" test code """
id_text = sentence_to_ids(vocab, sentence)
# id_text = [sentence_to_ids(vocab, sentence) for sentence in sentences]
print(f"sentence[0]: {sentence[0]}")
print(f"id_text[0]: {id_text[0]}")
""" test code """

def pad_seq(seq, max_length):
    """Padding function

    Args:
        seq (list of int): list of word index
        max_length (int): max length of the batch
    Return:
        seq (list of int): list of word index
    """
    seq += [PAD for i in range(max_length - len(seq))]
    return seq

# Hyper Parameters
batch_size = 64
n_batches = 500
vocab_size = len(vocab.word2id)
embedding_size = 300
