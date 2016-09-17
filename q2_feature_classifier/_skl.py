# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from itertools import islice
from collections import Counter

import skbio
from sklearn.externals.joblib import Parallel, delayed
import numpy

_specific_fitters = [
    ['svc', {'steps': [
     ['vectorize', 'sklearn.feature_extraction.DictVectorizer'],
     ['transform', 'sklearn.feature_selection.SelectPercentile'],
     ['classify', 'sklearn.svm.SVC']],
     'classify': {'C': 10, 'kernel': 'linear', 'degree': 3, 'gamma': 0.001}}],
    ['naive_bayes', {'steps': [
     ['vectorize', 'sklearn.feature_extraction.DictVectorizer'],
     ['classify', 'sklearn.naive_bayes.MultinomialNB']],
                     'classify': {'alpha': 0.01}}]]


def _extract_features(reads, word_length):
    return zip(*[_read_to_counts(read, word_length) for read in reads])


def _extract_labels(y, taxonomy_separator, taxonomy_depth, multioutput):
    labels = []
    for label in y:
        if taxonomy_separator != '':
            label = label.split(taxonomy_separator)
            if taxonomy_depth > 0:
                label = label[:taxonomy_depth]
            if not multioutput:
                label = taxonomy_separator.join(label)
        labels.append(label)
    return labels


def fit_pipeline(reads, taxonomy, pipeline, word_length=8,
                 taxonomy_separator='', taxonomy_depth=-1,
                 multioutput=False):
    seq_ids, X = _extract_features(reads, word_length)
    y = [taxonomy.get(s, 'unknown') for s in seq_ids]
    y = _extract_labels(y, taxonomy_separator, taxonomy_depth, multioutput)
    pipeline.fit(X, y)
    return pipeline


def _bootstrap_probs(counts):
    kmers, p = zip(*counts.items())
    p = numpy.array(p, dtype=float)
    p /= p.sum()
    return kmers, p


def _bootstrap(kmers, p, size):
    return Counter(numpy.random.choice(kmers, size=size, replace=True, p=p))


def _read_to_counts(read, word_length, confidence=-1.):
    if isinstance(read, skbio.DNA):
        seq_id = read.metadata['id']
        counts = read.kmer_frequencies(word_length)
        if confidence >= 0.:
            kmers, p = _bootstrap_probs(counts)
            size = len(read) // word_length
            bootstraps = [_bootstrap(kmers, p, size) for _ in [None]*100]
    else:
        seq_id = read[0].metadata['id']
        left = {k+'l': c
                for k, c in read[0].kmer_frequencies(word_length).items()}
        right = {k+'r': c
                 for k, c in read[1].kmer_frequencies(word_length).items()}
        counts = {**left, **right}
        if confidence >= 0.:
            lk, lp = _bootstrap_probs(left)
            rk, rp = _bootstrap_probs(right)
            ls, rs = [len(r) // word_length for r in read]
            bootstraps = [{**_bootstrap(lk, lp, ls), **_bootstrap(rk, rp, rs)}
                          for _ in [None]*100]
    if confidence < 0.:
        return seq_id, counts
    return seq_id, counts, bootstraps


def predict(reads, pipeline, word_length=None, taxonomy_separator='',
            taxonomy_depth=None, multioutput=False, chunk_size=262144,
            n_jobs=1, pre_dispatch='2*n_jobs', confidence=-1.):
    if confidence >= 0.:
        chunk_size //= 101
    return (m for c in Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)
            (delayed(_predict_chunk)(pipeline, multioutput, taxonomy_separator,
                                     word_length, confidence, chunk)
             for chunk in _chunks(reads, chunk_size)) for m in c)


def _predict_chunk(pipeline, multioutput, taxonomy_separator, word_length,
                   confidence, chunk):
    if confidence < 0.:
        return _predict_chunk_without_bs(pipeline, multioutput,
                                         taxonomy_separator, word_length,
                                         chunk)
    else:
        return _predict_chunk_with_bs(pipeline, multioutput,
                                      taxonomy_separator, word_length,
                                      confidence, chunk)


def _predict_chunk_without_bs(pipeline, multioutput, taxonomy_separator,
                              word_length, chunk):
    seq_ids, X = zip(*[_read_to_counts(read, word_length) for read in chunk])
    y = pipeline.predict(X)
    result = []
    for seq_id, taxon in zip(seq_ids, y):
        if multioutput:
            taxon = taxonomy_separator.join(taxon)
        result.append((seq_id, taxon, -1.))
    return result


def _predict_chunk_with_bs(pipeline, multioutput, taxonomy_separator,
                           word_length, confidence, chunk):
    seq_ids = []
    X = []
    for read in chunk:
        seq_id, count, bs = _read_to_counts(read, word_length, confidence)
        seq_ids.append(seq_id)
        X.append(count)
        X.extend(bs)
    y = pipeline.predict(X)
    result = []
    for seq_id, i in zip(seq_ids, range(0, len(y), 101)):
        labels = [label if multioutput else label.split(taxonomy_separator)
                  for label in y[i:i+101]]
        candidate_taxon = labels[0]
        confidences = Counter([l for label in labels[1:] for l in label])
        confidence *= 100
        taxon = []
        for level in candidate_taxon:
            if confidences[level] < confidence:
                break
            taxon.append(level)
            taxon_confidence = confidences[level]
        if len(taxon) == 0:
            continue
        taxon = taxonomy_separator.join(taxon)
        result.append((seq_id, taxon, taxon_confidence/100))
    return result


def _chunks(reads, chunk_size):
    while True:
        chunk = list(islice(reads, chunk_size))
        if len(chunk) == 0:
            break
        yield chunk
