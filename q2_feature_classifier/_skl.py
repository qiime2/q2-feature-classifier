# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from itertools import islice, repeat
from collections import Counter

import skbio
from sklearn.externals.joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
import numpy

_specific_fitters = [
    ['svc', {'steps': [
     ['vectorize', 'feature_extraction.DictVectorizer'],
     ['transform', 'feature_selection.SelectPercentile'],
     ['classify', 'svm.SVC']],
     'classify': {'C': 10, 'kernel': 'linear', 'degree': 3, 'gamma': 0.001}}],
    ['naive_bayes', {'steps': [
     ['hash', 'feature_extraction.FeatureHasher'],
     ['classify', 'naive_bayes.MultinomialNB']],
     'classify': {'alpha': 0.01},
     'hash': {'non_negative': True, 'n_features': 8192}}]]


def _extract_features(reads, word_length):
    return zip(*[_read_to_counts(read, word_length) for read in reads])


class MultioutputPipeline(object):
    # This is a hack because it looks like multioutput classifiers can't
    # handle non-numeric labels like regular classifiers.
    # TODO: raise issue linked to
    # https://github.com/scikit-learn/scikit-learn/issues/556

    def __init__(self, pipeline, taxonomy_separator):
        self._pipeline = pipeline
        self._separator = taxonomy_separator

    def fit(self, X, y):
        y = list(zip(*[l.split(self._separator) for l in y]))
        self._encoders = [LabelEncoder() for _ in range(len(y))]
        y = [e.fit_transform(l) for e, l in zip(self._encoders, y)]
        self._pipeline.fit(X, list(zip(*y)))

    def predict(self, X):
        y = self._pipeline.predict(X).astype(int)
        y = [e.inverse_transform(l) for e, l in zip(self._encoders, y.T)]
        return [self._separator.join(l) for l in zip(*y)]


def fit_pipeline(reads, taxonomy, pipeline, word_length=8,
                 taxonomy_separator='', taxonomy_depth=-1,
                 multioutput=False):
    seq_ids, X = _extract_features(reads, word_length)

    y = [taxonomy.get(s, 'unknown') for s in seq_ids]
    if taxonomy_depth > 0:
        for i, label in enumerate(y):
            label = label.split(taxonomy_separator)[:taxonomy_depth]
            y[i] = taxonomy_separator.join(label)
    if multioutput:
        pipeline = MultioutputPipeline(pipeline, taxonomy_separator)

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
            bootstraps = [_bootstrap(kmers, p, size) for _ in range(100)]
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
                          for _ in range(100)]
    if confidence < 0.:
        return seq_id, counts
    return seq_id, counts, bootstraps


def predict(reads, pipeline, word_length=None, taxonomy_separator='',
            taxonomy_depth=None, multioutput=False, chunk_size=262144,
            n_jobs=1, pre_dispatch='2*n_jobs', confidence=-1.):
    if confidence >= 0.:
        chunk_size = chunk_size // 101 + 1
    return (m for c in Parallel(n_jobs=n_jobs, batch_size=1,
                                pre_dispatch=pre_dispatch)
            (delayed(_predict_chunk)(pipeline, taxonomy_separator,
                                     word_length, confidence, chunk)
             for chunk in _chunks(reads, chunk_size)) for m in c)


def _predict_chunk(pipeline, taxonomy_separator, word_length,
                   confidence, chunk):
    if confidence < 0.:
        return _predict_chunk_without_bs(pipeline, taxonomy_separator,
                                         word_length, chunk)
    else:
        return _predict_chunk_with_bs(pipeline, taxonomy_separator,
                                      word_length, confidence, chunk)


def _predict_chunk_without_bs(pipeline, taxonomy_separator,
                              word_length, chunk):
    seq_ids, X = zip(*[_read_to_counts(read, word_length) for read in chunk])
    y = pipeline.predict(X)
    return zip(seq_ids, y, repeat(-1.))


def _predict_chunk_with_bs(pipeline, taxonomy_separator,
                           word_length, confidence, chunk):
    seq_ids = []
    X = []
    for read in chunk:
        seq_id, count, bs = _read_to_counts(read, word_length, confidence)
        seq_ids.append(seq_id)
        X.append(count)
        X.extend(bs)
    y = pipeline.predict(X)
    results = []
    confidence *= 100
    for seq_id, i in zip(seq_ids, range(0, len(y), 101)):
        taxon = y[i].split(taxonomy_separator)
        bootstraps = [l.split(taxonomy_separator) for l in y[i+1:i+101]]
        result = None
        for i in range(1, len(taxon)+1):
            matches = sum(taxon[:i] == bs[:i] for bs in bootstraps)
            if matches < confidence:
                break
            result = taxon[:i]
            result_confidence = matches/100
        if result is not None:
            result = taxonomy_separator.join(result)
            results.append((seq_id, result, result_confidence))
    return results


def _chunks(reads, chunk_size):
    while True:
        chunk = list(islice(reads, chunk_size))
        if len(chunk) == 0:
            break
        yield chunk
