# -*- coding: utf-8 -*-
__author__ = 'Brice Olivier'

import numpy as np
from metawordembedding import MetaWordEmbedding
from metawordembedding import WordEmbedding
import pytest


@pytest.fixture(scope="module")
def meta_word_embedding_init():
    we1 = WordEmbedding('../embeddings/wiki.fr.bin', '../embeddings/wiki.fr.vec', nmax=10000)
    we2 = WordEmbedding('../embeddings/cc.fr.300.bin', '../embeddings/cc.fr.300.vec', nmax=10000)
    return MetaWordEmbedding([we1, we2])

def test_merge_embeddings(meta_word_embedding_init):
    vocabulary1 = meta_word_embedding_init.word_embeddings[0].get_vocabulary()
    vocabulary2 = meta_word_embedding_init.word_embeddings[1].get_vocabulary()
    assert np.array_equal(np.sort(vocabulary1), np.sort(vocabulary2))
    assert meta_word_embedding_init.word_embeddings[0].vectors.shape ==\
           meta_word_embedding_init.word_embeddings[1].vectors.shape

def test_get_nn(meta_word_embedding_init):
    roi_vector1 = meta_word_embedding_init.word_embeddings[0].get_word_vector('roi')
    homme_vector1 = meta_word_embedding_init.word_embeddings[0].get_word_vector('homme')
    femme_vector1 = meta_word_embedding_init.word_embeddings[0].get_word_vector('femme')
    roi_vector2 = meta_word_embedding_init.word_embeddings[1].get_word_vector('roi')
    homme_vector2 = meta_word_embedding_init.word_embeddings[1].get_word_vector('homme')
    femme_vector2 = meta_word_embedding_init.word_embeddings[1].get_word_vector('femme')
    nn = meta_word_embedding_init.get_nn([roi_vector1 - homme_vector1 + femme_vector1,
                                          roi_vector2 - homme_vector2 + femme_vector2], k=5)
    assert len(nn) == 2
    assert len(nn[0]) == 5
    assert len(nn[1] == 5)
    assert nn[0][1] == 'reine'
