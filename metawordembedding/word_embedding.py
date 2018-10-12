# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import io
import numpy as np
import subprocess
import tempfile


class WordEmbedding():
    def __init__(self, model_path, embeddings_path, nmax=10000):
        self.model_path = model_path
        self.embeddings_path = embeddings_path
        self._load_vec(embeddings_path, nmax)

    def _load_vec(self, embeddings_path, nmax):
        """Load word vectors file and sets a numpy.matrix with vectors,
        an id2word dictionnary and a word2id dictionnary."""
        vectors = []
        word2id = {}
        with io.open(embeddings_path, 'r', encoding='utf-8',
                     newline='\n', errors='ignore') as f:
            next(f)
            for i, line in enumerate(f):
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                assert word not in word2id, 'word found twice'
                vectors.append(vect)
                word2id[word] = len(word2id)
                if len(word2id) == nmax:
                    break
        self.word2id = word2id
        self.id2word = {v: k for k, v in word2id.items()}
        self.vectors = np.vstack(vectors)

    def get_vocabulary(self):
        return list(self.word2id.keys())

    def is_word_in_vocabulary(self, word):
        return word in self.get_vocabulary()

    def get_word_vector(self, word):
        assert self.is_word_in_vocabulary(word), 'word not in dict'
        return self.vectors[self.word2id[word], :]

    def get_nn_given_vector(self, vector, k=1):
        """take as input a word vector and find its K nearest neighbors
        within a numpy.matrix of vectors and its id2word dictionnary."""
        scores = (self.vectors / np.linalg.norm(self.vectors, 2, 1)[:, None]).dot(
            vector / np.linalg.norm(vector))
        k_best = scores.argsort()[-k:][::-1]
        return [self.id2word[idx] for _, idx in enumerate(k_best)], scores[k_best]

    def get_nn_given_word(self, word, k=1):
        """take as input a word and find its K nearest neighbors
                within a numpy.matrix of vectors and its id2word dictionnary."""
        assert self.is_word_in_vocabulary(word), 'word not in dict'
        vector = self.vectors[self.word2id[word], :]
        scores = (self.vectors / np.linalg.norm(self.vectors, 2, 1)[:, None]).dot(
            vector / np.linalg.norm(vector))
        k_best = scores.argsort()[-k:][::-1]
        return [self.id2word[idx] for _, idx in enumerate(k_best)], scores[k_best]

    def get_vectors_for_oov_words(self, oov_words):
        _, oov_queries_path = tempfile.mkstemp()
        _, oov_embeddings_path = tempfile.mkstemp()
        with io.open(oov_queries_path, 'w', encoding='utf-8') as f:
            for w in oov_words:
                f.write(w + '\n')
        subprocess.call('../fastText-0.1.0/fasttext print-word-vectors ' +
                        self.model_path + '<' + oov_queries_path +
                        '>' + oov_embeddings_path, shell=True)
        vectors = []
        valid_oov_words = []
        with io.open(oov_embeddings_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split(" ")
                word = fields[0]
                vec = [float(v) for v in fields[1:]]
                if (len(vec) == self.vectors.shape[1]) and (np.linalg.norm(vec) != 0):
                    valid_oov_words.append(word)
                    vectors.append(vec)
        return (valid_oov_words, np.vstack(vectors))

    def add_words(self, words, vectors):
        for word in words:
            self.word2id[word] = len(self.word2id) - 1
            self.id2word[len(self.word2id) - 1] = word
        self.vectors = np.row_stack((self.vectors, vectors))