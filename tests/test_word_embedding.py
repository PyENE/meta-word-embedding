# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import numpy as np
from metawordembedding import WordEmbedding
import pytest


@pytest.fixture(scope="module")
def word_embedding_init():
    return WordEmbedding('../embeddings/wiki.fr.bin', '../embeddings/wiki.fr.vec', nmax=10000)


@pytest.fixture(scope="module")
def oov_embeddings_init(word_embedding_init):
    return word_embedding_init.get_embeddings_for_oov_words(['femm'])


def test_get_nn_given_word(word_embedding_init):
    nn = word_embedding_init.get_nn_given_word('appeler', k=3)
    assert len(nn) == 2
    assert len(nn[0]) == 3
    assert len(nn[1]) == 3
    assert np.array_equal(nn[0], ['appeler', 'appelle', 'nommer', 'rappeler', 'entendre'])
    assert np.array_equal(nn[1], [1., 0.61844597, 0.61445533, 0.61172119, 0.58956483])


def test_get_nn_given_embedding(word_embedding_init):
    roi_embedding = word_embedding_init.get_word_embedding('roi')
    homme_embedding = word_embedding_init.get_word_embedding('homme')
    femme_embedding = word_embedding_init.get_word_embedding('femme')
    nn = word_embedding_init.get_nn_given_embedding(roi_embedding - homme_embedding + femme_embedding, k=2)
    assert len(nn) == 2
    assert len(nn[0]) == 2
    assert len(nn[1]) == 2
    assert nn[0][1] == 'reine'

def test_get_embeddings_for_oov_words(oov_embeddings_init):
    assert oov_embeddings_init[1].shape[1] == 300

def test_add_words(word_embedding_init, oov_embeddings_init):
    init_embeddings_shape = word_embedding_init.embeddings.shape
    oov_embeddings_shape = oov_embeddings_init[1].shape
    word_embedding_init.add_words(oov_embeddings_init[0], oov_embeddings_init[1])
    complete_embeddings_shape = word_embedding_init.embeddings.shape
    assert len(np.setdiff1d(oov_embeddings_init[0], word_embedding_init.get_vocabulary())) == 0
    assert complete_embeddings_shape[0] == init_embeddings_shape[0] + oov_embeddings_shape[0]


