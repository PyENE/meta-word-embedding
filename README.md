# meta-word-embedding

## Installation

```bash
git clone https://github.com/brice-olivier/meta-word-embedding.git
cd meta-word-embedding
sudo python setup.py install --user
chmod +x ./bin/setup_fasttext.sh
./bin/setup_fasttext.sh
```

## If you do not already have fasttext models and embeddings (in french)

```bash
chmod +x ./bin/wget_cc.fr.300.sh
./bin/wget_cc.fr.300.sh
chmod +x ./bin/wget_wiki.fr.sh
./bin/wget_wiki.fr.sh
```

## Usage

```python
from metawordembedding import MetaWordEmbedding
from metawordembedding import WordEmbedding

we_wiki = WordEmbedding('./embeddings/wiki.fr.bin', './embeddings/wiki.fr.vec', nmax=10000)
we_cc = WordEmbedding('./embeddings/cc.fr.300.bin', './embeddings/cc.fr.300.vec', nmax=10000)
mwe = MetaWordEmbedding([we_wiki, we_cc])

roi_vector_wiki = mwe.word_embeddings[0].get_word_vector('roi')
homme_vector_wiki = mwe.word_embeddings[0].get_word_vector('homme')
femme_vector_wiki = mwe.word_embeddings[0].get_word_vector('femme')

roi_vector_cc = mwe.word_embeddings[1].get_word_vector('roi')
homme_vector_cc = mwe.word_embeddings[1].get_word_vector('homme')
femme_vector_cc = mwe.word_embeddings[1].get_word_vector('femme')

mwe.get_nn([roi_vector_wiki - homme_vector_wiki + femme_vector_wiki,
            roi_vector_cc - homme_vector_cc + femme_vector_cc], k=5)
```
Output:
```
(['roi', 'reine', 'rois', 'princesse', 'souverain'], array([0.86524063, 0.70102485, 0.60086826, 0.57122776, 0.55460754]))
```