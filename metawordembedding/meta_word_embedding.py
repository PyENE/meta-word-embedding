# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

from functools import reduce
import numpy as np


class MetaWordEmbedding():
    def __init__(self, word_embeddings):
        self.word_embeddings = word_embeddings
        self.n_embeddings = len(word_embeddings)
        self._merge_embeddings()

    def _merge_embeddings(self):
        vocabulary = reduce(np.union1d, [word_embedding.get_vocabulary() for word_embedding in self.word_embeddings])
        non_valid_oov_words = []
        for word_embedding in self.word_embeddings:
            oov_words = np.setdiff1d(vocabulary, word_embedding.get_vocabulary())
            (valid_oov_words, vectors) = word_embedding.get_vectors_for_oov_words(oov_words)
            non_valid_oov_words.append(np.setdiff1d(oov_words, valid_oov_words))
            word_embedding.add_words(valid_oov_words, vectors)
        non_valid_oov_words = reduce(np.union1d, non_valid_oov_words)
        valid_vocabulary = np.setdiff1d(vocabulary, non_valid_oov_words)
        for word_embedding in self.word_embeddings:
            id = [word_embedding.word2id[word] for word in valid_vocabulary]
            word_embedding.vectors = word_embedding.vectors [id]
            word_embedding.id2word = dict(zip(range(0, len(valid_vocabulary)), valid_vocabulary))
            word_embedding.word2id = {v: k for k, v in word_embedding.id2word.items()}

    def get_nn(self, vectors, k=1):
        assert len(vectors) == len(self.word_embeddings)
        scores = []
        for word_embedding, vector in zip(self.word_embeddings, vectors):
            scores.append((word_embedding.vectors / np.linalg.norm(word_embedding.vectors , 2, 1)[:, None]).dot(
                vector / np.linalg.norm(vector)))
        scores = np.stack(scores)
        scores = np.mean(scores, 0)
        k_best = scores.argsort()[-k:][::-1]
        return [self.word_embeddings[0].id2word[idx] for _, idx in enumerate(k_best)], scores[k_best]
