"""
Examples of use:
$ python benchmark-word-embeddings.py -a models/wiki.fr.bin -c embeddings/wiki.fr.vec -q benchmarks/questions-words-fr.txt -o results/wiki-benchmark.txt -m exception -v 100000
$ python benchmark-word-embeddings.py -a models/wiki.fr.bin -b models/cc.fr.300.bin -c embeddings/wiki.fr.vec -d cc.fr.300.vec -q benchmarks/questions-words-fr.txt -o results/aggregated-benchmark.txt -m top -t 1 -v 100000
"""

from __future__ import print_function
import io
import numpy as np
from optparse import OptionParser
import subprocess
import sys
import tempfile
import time


def load_vec(emb_path, nmax=50000):
    """Load word embeddings file and returns a numpy.matrix with embeddings,
    an id2word dictionnary and a word2id dictionnary."""
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8',
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
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


def get_nn(word_embedding, embeddings, id2word, k=1):
    """take as input a word embedding and find its K nearest neighbors
    within a numpy.matrix of embeddings and its id2word dictionnary."""
    scores = (embeddings / np.linalg.norm(embeddings, 2, 1)[:, None]).dot(
        word_embedding / np.linalg.norm(word_embedding))
    k_best = scores.argsort()[-k:][::-1]
    return [id2word[idx] for _, idx in enumerate(k_best)], scores[k_best]


def get_custom_nn(word_embedding1, word_embedding2, embeddings1, embeddings2, id2word, k=1):
    """embeddings1 and embeddings2 must have shared id2word
    """
    scores1 = (embeddings1 / np.linalg.norm(embeddings1, 2, 1)[:, None]).dot(
        word_embedding1 / np.linalg.norm(word_embedding1))
    scores2 = (embeddings2 / np.linalg.norm(embeddings2, 2, 1)[:, None]).dot(
        word_embedding2 / np.linalg.norm(word_embedding2))
    scores = (scores1 + scores2) / 2
    k_best = scores.argsort()[-k:][::-1]
    return [id2word[idx] for _, idx in enumerate(k_best)], scores[k_best]


def evaluate_predicted_embedding(embeddings, id2word, word2id, question, method="top", k=1):
    """test"""
    if method == "exception":
        k = 2
    if type(embeddings) is tuple:
        pred_embedding1 = embeddings[0][word2id[question[1]]] - \
            embeddings[0][word2id[question[0]]] + \
            embeddings[0][word2id[question[2]]]
        pred_embedding2 = embeddings[1][word2id[question[1]]] - \
            embeddings[1][word2id[question[0]]] + \
            embeddings[1][word2id[question[2]]]
        nn, _ = get_custom_nn(pred_embedding1, pred_embedding2,
                              embeddings[0], embeddings[1], id2word, k)
    else:
        pred_embedding = embeddings[word2id[question[1]]] - \
            embeddings[word2id[question[0]]] + \
            embeddings[word2id[question[2]]]
        nn, _ = get_nn(pred_embedding, embeddings, id2word, k)
    nn = np.array(nn)
    # print("question ^ top predictions: %s - %s + %s = %s ^ %s" %
    #       (question[1], question[0], question[2], question[3], nn))
    if method == "top":
        return question[3] in nn
    elif method == "exception":
        return question[3] == nn[nn != question[2]][0]


def load_benchmark(benchmark_path):
    """Load word embeddings file and returns a numpy.matrix with embeddings,
    an id2word dictionnary and a word2id dictionnary."""
    questions_words = {}
    preset_results = {}
    with io.open(benchmark_path, 'r', encoding='utf-8',
                 newline='\n', errors='ignore') as f:
        for _, line in enumerate(f):
            line = line.split("\n")[0]
            if ":" in line:
                key = line.rstrip().split(":")[1][1:]
                questions_words[key] = []
                preset_results[key] = []
            else:
                question = line.rstrip().split(" ")
                assert(len(question) == 4)
                questions_words[key].append(question)
                preset_results[key].append(0)
    return questions_words, preset_results


def compute_embeddings_for_oov(embeddings, oov_words, id2word, word2id, model_path):
    _, oov_words_queries_file_name = tempfile.mkstemp()
    _, oov_words_file_name = tempfile.mkstemp()
    print("model %s" % model_path)
    print("oov words: %s" % oov_words_queries_file_name)
    print("oov words embeddings: %s" % oov_words_file_name)
    with io.open(oov_words_queries_file_name, 'w', encoding='utf-8') as f:
        for w in oov_words:
            f.write(w + '\n')
    subprocess.call('./fastText-0.1.0/fasttext' +
                    ' print-word-vectors ' +
                    model_path +
                    '<' +
                    oov_words_queries_file_name +
                    '>' + oov_words_file_name,
                    shell=True)
    vectors = []
    with io.open(oov_words_file_name, 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split(" ")
            word = fields[0]
            vec = [float(v) for v in fields[1:]]
            if len(vec) != embeddings.shape[1]:
                # print("Warning: embedding size %d, with field[0] %s, line starting with %s" %
                #       (len(vec), word, line[0:10]), file=sys.stderr)
                pass
            elif np.linalg.norm(vec) == 0:
                # print("Warning: embedding for word %s has norm 0" % word, file=sys.stderr)
                pass
            else:
                word2id[word] = len(word2id) - 1
                id2word[len(word2id) - 1] = word
                vectors.append(vec)
    embeddings = np.row_stack((embeddings, np.vstack(vectors)))
    return embeddings, id2word, word2id


def run_benchmark(questions_words, preset_results, embeddings, id2word, word2id, output_path, method="top", k=3):
    t0 = time.time()
    for key in questions_words:
        print("\n____________________________________________________")
        print("Questions-Words category%s" % key)
        i = 0
        t1 = time.time()
        for question in questions_words[key]:
            preset_results[key][i] = evaluate_predicted_embedding(
                embeddings, id2word, word2id, question, method, k)
            sys.stdout.write("\rProgress: %.4f\tAccuracy: %.4f" %
                             (float(i + 1) / len(preset_results[key]),
                              float(np.sum(preset_results[key])) / (i + 1)))
            sys.stdout.flush()
            i += 1
        t2 = time.time()
        with open(output_path, 'a') as f:
            f.write("____________________________________________________\n")
            f.write("Questions-Words category%s\n" % key)
            f.write("Accuracy: %.4f\n" %
                    (float(np.sum(preset_results[key])) /
                     len(preset_results[key])))
            f.write("Time: %.0fs\n" % (t1 - t0))
        print('\nTime: %.4fs' % (t2 - t1))
    t3 = time.time()
    overall_accuracy = float(np.sum([np.sum(preset_results[key])
                                     for key in questions_words])) / np.sum(
                                         [len(preset_results[key])
                                          for key in questions_words])
    overall_time = t3 - t0
    print("____________________________________________________")
    print("Overall accuracy %.4f\n" % overall_accuracy)
    print("Overall time %.0f\n" % overall_time)
    with open(output_path, 'a') as f:
        f.write("____________________________________________________\n")
        f.write("Overall accuracy %.4f\n" % overall_accuracy)
        f.write("Overall time %.0f\n" % overall_time)


parser = OptionParser()

parser.add_option("-a", "--model1", dest="model_path1",
                  help="Path to the model1 with embedding (.bin).",
                  default=None)

parser.add_option("-b", "--model2", dest="model_path2",
                  help="Path to the model2 with embedding (.bin).",
                  default=None)

parser.add_option("-c", "--embeddings1", dest="embeddings_path1",
                  help="Path to the embeddings1 (.vec).",
                  default=None)

parser.add_option("-d", "--embeddings2", dest="embeddings_path2",
                  help="Path to the embeddings2 (.vec).",
                  default=None)

parser.add_option("-q", "--questions", dest="questions_path",
                  help="Path to the benchmark questions (.txt).",
                  default=None)

parser.add_option("-o", "--output", dest="output_path",
                  help="Path to the benchmark output (.txt).",
                  default=None)

parser.add_option("-v", "--vocabulary_size", dest="vocabulary_size",
                  help="Size of the vocabulary to import from .vec (int).",
                  default=100000)

parser.add_option("-m", "--method", dest="method",
                  help="method to evaluate predicted embedding"
                  "(\"top\" or \"exception\").",
                  default="top")

parser.add_option("-t", "--top", dest="top",
                  help="argument for top method (int)",
                  default=3)

if __name__ == "__main__":
    (options, args) = parser.parse_args()
    options.top = int(options.top)
    options.vocabulary_size = int(options.vocabulary_size)
    print("loading benchmark... ", end="")
    questions_words, preset_results = load_benchmark(options.questions_path)
    print("ok\nloading embeddings1... ", end="")
    embeddings1, id2word1, word2id1 = load_vec(options.embeddings_path1,
                                               options.vocabulary_size)
    print("ok")
    word_list = np.unique([item for sublist in questions_words.values()
                           for subsublist in sublist for item in subsublist])
    oov_words1 = np.setdiff1d(word_list, id2word1.values())
    print("%d question words not in embeddings1" % len(oov_words1))
    if all(v is not None for v in [options.model_path2, options.embeddings_path2]):
        print("loading embeddings2... ", end="")
        embeddings2, id2word2, word2id2 = load_vec(options.embeddings_path2,
                                                   options.vocabulary_size)
        print("ok")
        # embeddings2 = embeddings2[id2word1.keys()]
        # id2word2 = id2word1
        # word2id2 = word2id1
        words_embeddings_1 = word2id1.keys()
        words_embeddings_2 = word2id2.keys()
        oov_words1 = np.concatenate((
            oov_words1, np.setdiff1d(words_embeddings_2, words_embeddings_1)))
        oov_words2 = np.setdiff1d(word_list, words_embeddings_2)
        print("%d question words not in embeddings2" % len(oov_words2))
        oov_words2 = np.concatenate((
            oov_words2, np.setdiff1d(words_embeddings_1, words_embeddings_2)))
        print("computing oov words for embeddings2... ", end="")
        embeddings2, id2word2, word2id2 = compute_embeddings_for_oov(
            embeddings2, oov_words2, id2word2, word2id2, options.model_path2)
        print("ok\ncomputing oov words for embeddings1... ", end="")

    embeddings1, id2word1, word2id1 = compute_embeddings_for_oov(
        embeddings1, oov_words1, id2word1, word2id1, options.model_path1)
    print("ok")
    if all(v is not None for v in [options.model_path2, options.embeddings_path2]):
        words = np.intersect1d(id2word1.values(), id2word2.values())
        id1 = [word2id1[w] for w in words]
        id2 = [word2id2[w] for w in words]
        embeddings1 = embeddings1[id1]
        embeddings2 = embeddings2[id2]
        id2word = dict(zip(range(0, len(words)), words))
        word2id = {v: k for k, v in id2word.items()}
        print("running benchmark...")
        run_benchmark(questions_words, preset_results,
                      (embeddings1, embeddings2), id2word, word2id,
                      options.output_path, options.method, options.top)
    else:
        run_benchmark(questions_words, preset_results, embeddings1, id2word1,
                      word2id1, options.output_path, options.method,
                      options.top)
