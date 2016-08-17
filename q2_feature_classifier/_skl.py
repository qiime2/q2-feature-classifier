# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import json

import skbio
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

_specific_fitters = [
        ('naive_bayes', [('s', SelectPercentile()), ('c', MultinomialNB())]),
        ('svc', [('s', SelectPercentile()),
                 ('c', SVC(C=10, kernel='linear', degree=3, gamma=0.001))])
        ]


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


def fit_pipeline(reads, taxonomy, spec, word_length, taxonomy_separator=None,
                 taxonomy_depth=None, multioutput=False):
    seq_ids, X = _extract_features(reads, word_length)
    y = [taxonomy.get(s, 'unknown') for s in seq_ids]
    y = _extract_labels(y, taxonomy_separator, taxonomy_depth, multioutput)
    pipeline = Pipeline(steps=spec['steps'])
    pipeline.set_params(**spec)
    pipeline.fit(X, y)
    return pipeline


def predict(reads, pipeline, word_length=None, taxonomy_separator=None,
            taxonomy_depth=None, multioutput=False):
    seq_ids, X = _extract_features(reads, word_length)
    y = pipeline.predict(X)
    if multioutput:
        y = [taxonomy_separator.join(t) for t in y]
    return seq_ids, y



