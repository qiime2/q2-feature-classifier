# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from itertools import islice, repeat
from copy import deepcopy

from sklearn.externals.joblib import Parallel, delayed
from q2_types.feature_data import PairedDNAIterator

#    ['svc', {'steps': [
#     ['feat_ext', 'feature_extraction.text.CountVectorizer'],
#     ['transform', 'feature_selection.SelectPercentile'],
#     ['classify', 'svm.SVC']],
#     'classify': {'C': 10, 'kernel': 'linear', 'degree': 3, 'gamma': 0.001},
#     'feat_ext': {'ngram_range': [8, 8], 'analyzer': 'char_wb'}}],

_specific_fitters = [
        ['naive_bayes',
         [['feat_ext',
           {'__type__': 'feature_extraction.text.HashingVectorizer',
            'analyzer': 'char_wb',
            'n_features': 8192,
            'ngram_range': [8, 8],
            'non_negative': True}],
          ['classify',
           {'__type__': 'custom.LowMemoryMultinomialNB',
            'alpha': 0.01}]]]]


def fit_pipeline(reads, taxonomy, pipeline):
    seq_ids, X = _extract_reads(reads, isinstance(reads, PairedDNAIterator))
    data = [(taxonomy[s], x) for s, x in zip(seq_ids, X) if s in taxonomy]
    y, X = list(zip(*data))
    pipeline.fit(X, y)
    return pipeline


def _extract_reads(reads, paired_end):
    if paired_end:
        return zip(*[(r[0].metadata['id'], b' '.join(e._string for e in r))
                     for r in reads])
    return zip(*[(r.metadata['id'], r._string) for r in reads])


def predict(reads, pipeline, separator=';', chunk_size=262144, n_jobs=1,
            pre_dispatch='2*n_jobs', confidence=-1.):
    if confidence >= 0.:
        chunk_size = chunk_size // 101 + 1
    paired_end = isinstance(reads, PairedDNAIterator)
    return (m for c in Parallel(n_jobs=n_jobs, batch_size=1,
                                pre_dispatch=pre_dispatch)
            (delayed(_predict_chunk)(pipeline, separator, paired_end,
                                     confidence, chunk)
             for chunk in _chunks(reads, chunk_size)) for m in c)


def _predict_chunk(pipeline, separator, paired_end, confidence, chunk):
    if confidence < 0.:
        return _predict_chunk_without_conf(pipeline, paired_end, chunk)
    else:
        return _predict_chunk_with_conf(pipeline, separator, paired_end,
                                        confidence, chunk)


def _predict_chunk_without_conf(pipeline, paired_end, chunk):
    seq_ids, X = _extract_reads(chunk, paired_end)
    y = pipeline.predict(X)
    return zip(seq_ids, y, repeat(-1.))


def _predict_chunk_with_conf(pipeline, separator, paired_end,
                             confidence, chunk):
    seq_ids, X = _extract_reads(chunk, paired_end)

    if not hasattr(pipeline, "predict_proba"):
        raise ValueError('this classifier does not support confidence values')
    prob_pos = pipeline.predict_proba(X)
    if prob_pos.shape != (len(X), len(pipeline.classes_)):
        raise ValueError('this classifier does not support confidence values')

    y = pipeline.classes_[prob_pos.argmax(axis=1)]

    results = []
    split_classes = [c.split(separator) for c in pipeline.classes_]
    for seq_id, taxon, class_probs in zip(seq_ids, y, prob_pos):
        taxon = taxon.split(separator)
        classes = zip(deepcopy(split_classes), class_probs)
        result = []
        for level in taxon:
            classes = [cls for cls in classes if cls[0].pop(0) == level]
            cum_prob = sum(c[1] for c in classes)
            if cum_prob < confidence:
                break
            result.append(level)
            result_confidence = cum_prob
        if len(result) > 0:
            result = separator.join(result)
            results.append((seq_id, result, result_confidence))

    return results


def _chunks(reads, chunk_size):
    while True:
        chunk = list(islice(reads, chunk_size))
        if len(chunk) == 0:
            break
        yield chunk
