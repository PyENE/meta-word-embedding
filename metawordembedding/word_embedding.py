# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import io
import numpy as np
import subprocess
import tempfile


class WordEmbedding():
    def __init__(self, model_path, embedding_path, nmax=10000):
        self.model_path = model_path
        self.embedding_path = embedding_path
        self._load_vec(embedding_path, nmax)

    def _load_vec(self, embedding_path, nmax):
        """Load word embeddings file and sets a numpy.matrix with embeddings,
        an id2word dictionnary and a word2id dictionnary."""
        vectors = []
        word2id = {}
        with io.open(embedding_path, 'r', encoding='utf-8',
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
        self.embeddings = np.vstack(vectors)

    def is_word_in_vocabulary(self, word):
        return word in self.word2id.keys()

    def get_word_embedding(self, word):
        assert self.is_word_in_vocabulary(word), 'word not in dict'
        return self.embeddings[self.word2id[word], :]

    def get_nn_given_embedding(self, word_embedding, k=1):
        """take as input a word embedding and find its K nearest neighbors
        within a numpy.matrix of embeddings and its id2word dictionnary."""
        scores = (self.embeddings / np.linalg.norm(self.embeddings, 2, 1)[:, None]).dot(
            word_embedding / np.linalg.norm(word_embedding))
        k_best = scores.argsort()[-k:][::-1]
        return [self.id2word[idx] for _, idx in enumerate(k_best)], scores[k_best]

    def get_nn_given_word(self, word, k=1):
        """take as input a word and find its K nearest neighbors
                within a numpy.matrix of embeddings and its id2word dictionnary."""
        assert self.is_word_in_vocabulary(word), 'word not in dict'
        word_embedding = self.embeddings[self.word2id[word], :]
        scores = (self.embeddings / np.linalg.norm(self.embeddings, 2, 1)[:, None]).dot(
            word_embedding / np.linalg.norm(word_embedding))
        k_best = scores.argsort()[-k:][::-1]
        return [self.id2word[idx] for _, idx in enumerate(k_best)], scores[k_best]

    def get_embeddings_for_oov(self, oov_words):
        _, oov_queries_path = tempfile.mkstemp()
        _, oov_embeddings_path = tempfile.mkstemp()
        with io.open(oov_queries_path, 'w', encoding='utf-8') as f:
            for w in oov_words:
                f.write(w + '\n')
        subprocess.call('../fastText-0.1.0/fasttext print-word-vectors ' +
                        self.model_path + '<' + oov_queries_path +
                        '>' + oov_embeddings_path, shell=True)
        vectors = []
        with io.open(oov_embeddings_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split(" ")
                vec = [float(v) for v in fields[1:]]
                vectors.append(vec)
        return np.vstack(vectors)

    def add_words_to_embeddings(self, words, embeddings):
        for word in words:
            self.word2id[word] = len(self.word2id) - 1
            self.id2word[len(self.word2id) - 1] = word
        self.embeddings = np.row_stack((self.embeddings, embeddings))