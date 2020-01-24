# ----------------------------------------------------------------------------
# Copyright (c) 2016-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from itertools import islice

import numpy
from scipy.sparse import vstack
from sklearn.base import BaseEstimator, ClassifierMixin, clone  # noqa
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted  # noqa
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer


class LowMemoryMultinomialNB(MultinomialNB):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None,
                 chunk_size=20000):
        self.chunk_size = chunk_size
        super().__init__(alpha=alpha, fit_prior=fit_prior,
                         class_prior=class_prior)

    def fit(self, X, y, sample_weight=None):
        if self.chunk_size <= 0:
            return super().fit(X, y, sample_weight=sample_weight)

        classes = numpy.unique(y)
        for i in range(0, X.shape[0], self.chunk_size):
            upper = min(i+self.chunk_size, X.shape[0])
            cX = X[i:upper]
            cy = y[i:upper]
            if sample_weight is None:
                csample_weight = None
            else:
                csample_weight = sample_weight[i:upper]
            self.partial_fit(cX, cy, sample_weight=csample_weight,
                             classes=classes)

        return self


class ChunkedHashingVectorizer(HashingVectorizer):
    # This class is a kludge to get around
    # https://github.com/scikit-learn/scikit-learn/issues/8941
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word', n_features=(2 ** 20),
                 binary=False, norm='l2', alternate_sign=True,
                 dtype=numpy.float64, chunk_size=20000):
        self.chunk_size = chunk_size
        super().__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, analyzer=analyzer, n_features=n_features,
            binary=binary, norm=norm, alternate_sign=alternate_sign,
            dtype=dtype)

    def transform(self, X):
        if self.chunk_size <= 0:
            return super().transform(X)

        returnX = None
        X = iter(X)
        while True:
            cX = list(islice(X, self.chunk_size))
            if len(cX) == 0:
                break
            cX = super().transform(cX)
            if returnX is None:
                returnX = cX
            else:
                returnX = vstack([returnX, cX])

        return returnX

    fit_transform = transform


# Experimental feature. USE WITH CAUTION
class _MultioutputClassifier(BaseEstimator, ClassifierMixin):
    # This is a hack because it looks like multioutput classifiers can't
    # handle non-numeric labels like regular classifiers.
    # TODO: raise issue linked to
    # https://github.com/scikit-learn/scikit-learn/issues/556

    def __init__(self, base_estimator=None, separator=';'):
        self.base_estimator = base_estimator
        self.separator = separator

    def fit(self, X, y, **fit_params):
        y = list(zip(*[l.split(self.separator) for l in y]))
        self.encoders_ = [LabelEncoder() for _ in y]
        y = [e.fit_transform(l) for e, l in zip(self.encoders_, y)]
        self.base_estimator.fit(X, list(zip(*y)), **fit_params)
        return self

    @property
    def classes_(self):
        classes = [e.inverse_transform(l) for e, l in
                   zip(self.encoders_, zip(*self.base_estimator.classes_))]
        return [self.separator.join(l) for l in zip(*classes)]

    def predict(self, X):
        y = self.base_estimator.predict(X).astype(int)
        y = [e.inverse_transform(l) for e, l in zip(self.encoders_, y.T)]
        return [self.separator.join(l) for l in zip(*y)]

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)
