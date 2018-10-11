# Directory content
## This directory does not contain:
* models/cc.fr.300.bin: wiki+cc+cbow model bin (2018)
downloadable here: https://fasttext.cc/docs/en/crawl-vectors.html
* embeddings/cc.fr.300.vec: representations of word vectors for wiki+cc+cbow (2018)
downloadable here: https://fasttext.cc/docs/en/crawl-vectors.html
* models/wiki.fr.bin: wiki+skipgram model bin (2016)
downloadable here: https://fasttext.cc/docs/en/pretrained-vectors.html
* embeddings/wiki.fr.vec: representations of word vectors for wiki+skipgram (2016)
downloadable here: https://fasttext.cc/docs/en/pretrained-vectors.html
* multilingual_embeddings.fr.txt: more embeddings in french (different model)

## This directory contains:

* fastText-0.1.0: source code for running fasttext algorithms:
https://github.com/facebookresearch/fastText
* queries/eighteen_oov_words_queries.txt: out of vocabulary words in cc.fr.300.vec
* queries/eighteen_oov_words.txt: out of vocabulary word representations using cc.fr.300.bin
* queries/sixteen_oov_words_queries.txt: out of vocabulary words in wiki.fr.vec
* queries/sixteen_oov_words.txt: out of vocabulary word representations using wiki.fr.bin
* benchmarks/questions-words-fr.txt: Analogy test in French
* benchmark-word-embeddings.py: benchmark pre-trained vectors on
questions-words-fr.txt benchmark. Used to benchmark cc18, wiki16 and their
aggregation.
* word-embeddings-benchmarks: a github package which evaluates pretrained vectors on 4 different tasks:
github repo: https://github.com/kudkudak/word-embeddings-benchmarks
associated paper: https://arxiv.org/pdf/1702.02170.pdf
benchmarking tasks: http://www.aclweb.org/anthology/D15-1036

## Analysis scripts:
* ema/notebook/cos_inst_with_fasttext.ipynb: add new columns in em-y35-fasttext.xlsx
with the cos inst from 2016, 2018 models and a "meta" model combining both 2016 and 2018 representations.

# Why Fasttext ?

LSA + corpus LeMonde:

## Preprocessing issues

"L'économie" has cos inst 0. Because of the cap ?
15	s21	35	prix_petrole-f1	0	1	236	353.4	280.8	173	0	78.5397988284	1.8240898324	1	1	0	1	1	1	L'économie~0	L'économie~0	15.74	0	0	f	forward	1

"Tokyo" has no cos inst -> proper nouns have no cos inst.
15	s21	50	hausse_bourse-f1	0	2	377	411.2	258.9	170	0	140.4197279587	-29.7600643492	1	0	0	2	9	45	Tokyo~3	de~2_Tokyo~3				f	upward	9


## Corpus issues

"brut" has low cos inst.
15	s21	35	prix_petrole-f1	0	3	731	539.4	320.2	207	0	91.8487887781	-24.1651705241	1	0	0	1	1	1	brut~9	pétrole~8_brut~9	3.31	0.09	0.02	f	downward	1


## Weak word additivity

"refugies" has low cos inst with "aide refugies"
15	s21	28	aide_refugies-f2	1	3	581	562.9	335.8	165	0	102.3308848784	20.8968144526	1	0	0	-2	-8	-44	réfugiés~13	de~10_réfugiés~13	8.18	0.06	-0.05	f	downward	0
inspection:
sim_cos(réfugier, aide) = 0.03
sim_cos(aide+réfugier,réfugier) = 0.06
sim_cos(aide+réfugier,aide) = 1.00
norm(aide) = 0.68
norm(réfugier) = 0.02


## Methodology-related limitations

"naturels" has high cos inst though it is a text A. LSA limitations. Temporal LDA as information acquisition model over time.
1	s01	1	chasse_oiseaux-a1	1	16	3287	324.1	474	183	0	38.2423848629	177.3021943667	1	0	0	0	0	0	naturels~19	naturels~19	5.88	0.43	-0.01	a	downward	0




## Fasttext:
Research directed by Tomas Mikolov (previously Google), notably known for word2vec in 2013.

Goal: represent word analogies and semantic

fasttext provides 2 models: skipgram and continuous bag of words (cbow)

skipgram model learns to predict the target using a random close-by word.
cbow model predicts the target word according to its context.
The context is represented as a bag of the words contained in a fixed size
window around the target word.

In practice, they observe that skipgram models works better with subword information than cbow. (Not sure it's still valid in 2018)

params:
On word dimension: By default, we use 100 dimensions, but any value in the 100-300 range is as popular.
The epoch parameter controls how many time will loop over your data.

nearest neighbor: assess type of semantic information the vector is able to capture
with cosine similarity

word analogies: king - man + woman = queen

importance of n-grams:
if a new word in unknown it'll still show its close-by words thanks to low-level semantic integration (syllable)
example: "skippeur" is out of vocabulary (OOV) but can be affected to its Nearest Neighbor (NN): "skipper".

word representations are augmented using
character ngrams. A vector representation is associated to
each character ngram, and the vector representation of a
word is obtained by taking the sum of the vectors of the
character ngrams appearing in the word. The full word is
always included as part of the character ngrams, so that the
model still learns one vector for each word.

Dealing with compounds (mots composés):
Compounds are also OOV. In practice, I've found good results by summing the words.
example: Treat "Haute-Corse" as "Haute" + "Corse"


## References:
* Grave 2018
* Bojanowki 2017
* Joulin 2016b
* Joulin 2016a
* Mikolov 2013b
* Mikolov 2013a
* https://fasttext.cc/docs/en/unsupervised-tutorial.html (fasttext tutorial)
* https://fasttext.cc/docs/en/pretrained-vectors.html (2016, 300 dim, skip-grams)
* https://fasttext.cc/docs/en/crawl-vectors.html
(2018, trained using CBOW with position-weights, in dimension 300,
with character n-grams of length 5, a window of size 5 and 10 negatives)
* https://www.quora.com/What-is-the-main-difference-between-word2vec-and-fastText
