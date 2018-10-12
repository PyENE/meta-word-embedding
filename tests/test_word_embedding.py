# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import numpy as np
from metawordembedding import WordEmbedding
import pytest


@pytest.fixture(scope="module")
def word_embedding_init():
    return WordEmbedding('../embeddings/wiki.fr.bin', '../embeddings/wiki.fr.vec', nmax=10000)


@pytest.fixture(scope="module")
def oov_vectors_init(word_embedding_init):
    return word_embedding_init.get_vectors_for_oov_words(['femm'])


def test_get_nn_given_word(word_embedding_init):
    nn = word_embedding_init.get_nn_given_word('appeler', k=3)
    assert len(nn) == 2
    assert len(nn[0]) == 3
    assert len(nn[1]) == 3
    assert np.array_equal(nn[0], ['appeler', 'appelle', 'nommer'])
    np.testing.assert_array_almost_equal(nn[1], np.array([1., 0.61844597, 0.61445533]))


def test_get_nn_given_embedding(word_embedding_init):
    roi_vector = word_embedding_init.get_word_vector('roi')
    homme_vector = word_embedding_init.get_word_vector('homme')
    femme_vector = word_embedding_init.get_word_vector('femme')
    nn = word_embedding_init.get_nn_given_vector(roi_vector - homme_vector + femme_vector, k=2)
    assert len(nn) == 2
    assert len(nn[0]) == 2
    assert len(nn[1]) == 2
    assert nn[0][1] == 'reine'

def test_get_vectors_for_oov_words(oov_vectors_init):
    assert oov_vectors_init[1].shape[1] == 300

def test_add_words(word_embedding_init, oov_vectors_init):
    init_vectors_shape = word_embedding_init.vectors.shape
    oov_vectors_shape = oov_vectors_init[1].shape
    word_embedding_init.add_words(oov_vectors_init[0], oov_vectors_init[1])
    complete_vectors_shape = word_embedding_init.vectors.shape
    assert len(np.setdiff1d(oov_vectors_init[0], word_embedding_init.get_vocabulary())) == 0
    assert complete_vectors_shape[0] == init_vectors_shape[0] + oov_vectors_shape[0]


