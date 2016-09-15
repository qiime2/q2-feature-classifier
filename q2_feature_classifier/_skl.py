# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from itertools import islice

import skbio
from joblib import Parallel, delayed

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


def _read_to_counts(read, word_length):
    if isinstance(read, skbio.DNA):
        return read.metadata['id'], read.kmer_frequencies(word_length)
    else:
        return (read[0].metadata['id'],
                {w+l: c for l, r in zip('lr', read)
                 for w, c in r.kmer_frequencies(word_length).items()})


def predict(reads, pipeline, word_length=None, taxonomy_separator=None,
            taxonomy_depth=None, multioutput=False, chunk_size=262144,
            n_jobs=1, pre_dispatch='2*n_jobs'):
    return (m for c in Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)
            (delayed(_predict_chunk)(pipeline, multioutput, taxonomy_separator,
                                     word_length, chunk) for chunk in
            _chunks(reads, chunk_size)) for m in c)


def _predict_chunk(pipeline, multioutput, taxonomy_separator, seq_ids,
                   word_length, chunk):
    seq_ids, X = zip(*[_read_to_counts(read, word_length) for read in chunk])
    y = pipeline.predict(X)
    for seq_id, taxon in zip(seq_ids, y):
        if multioutput:
            taxon = taxonomy_separator.join(taxon)
        yield seq_id, taxon


def _chunks(reads, chunk_size):
    while True:
        chunk = list(islice(reads, chunk_size))
        if len(chunk) == 0:
            break
        yield chunk
