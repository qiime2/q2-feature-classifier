# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import skbio
from sklearn.feature_extraction import DictVectorizer

_specific_fitters = [
    ['svc', {'steps': [
     ['transform', 'sklearn.feature_selection.SelectPercentile'],
     ['classify', 'sklearn.svm.SVC']],
     'classify': {'C': 10, 'kernel': 'linear', 'degree': 3, 'gamma': 0.001}}],
    ['naive_bayes', {'steps': [
     ['transform', 'sklearn.feature_selection.SelectPercentile'],
     ['classify', 'sklearn.naive_bayes.MultinomialNB']]}]]


def _extract_features(reads, word_length):
    seq_ids = []
    counts = []
    for read in reads:
        if isinstance(read, skbio.DNA):
            seq_ids.append(read.metadata['id'])
            counts.append(read.kmer_frequencies(word_length))
        else:
            seq_ids.append(read[0].metadata['id'])
            count = {w+l: c for l, r in zip('lr', read)
                     for w, c in r.kmer_frequencies(word_length).items()}
            counts.append(count)
    vectoriser = DictVectorizer()
    return seq_ids, vectoriser.fit_transform(counts)


def _extract_labels(y, taxonomy_separator, taxonomy_depth, multioutput):
    labels = []
    for label in y:
        if taxonomy_separator is not None:
            label = label.split(taxonomy_separator)
            if taxonomy_depth is not None:
                label = label[:taxonomy_depth]
            if not multioutput:
                label = taxonomy_separator.join(label)
        labels.append(label)
    return labels


def fit_pipeline(reads, taxonomy, pipeline, word_length=8,
                 taxonomy_separator=None, taxonomy_depth=None,
                 multioutput=False):
    seq_ids, X = _extract_features(reads, word_length)
    y = [taxonomy.get(s, 'unknown') for s in seq_ids]
    y = _extract_labels(y, taxonomy_separator, taxonomy_depth, multioutput)
    pipeline.fit(X, y)
    return pipeline


def predict(reads, pipeline, word_length=None, taxonomy_separator=None,
            taxonomy_depth=None, multioutput=False):
    seq_ids, X = _extract_features(reads, word_length)
    y = pipeline.predict(X)
    if multioutput:
        y = [taxonomy_separator.join(t) for t in y]
    return seq_ids, y
