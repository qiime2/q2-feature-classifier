# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import numpy
from sklearn.base import BaseEstimator, ClassifierMixin, clone  # noqa
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted  # noqa
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder


class LowMemoryMultinomialNB(MultinomialNB):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None,
                 chunk_size=-1):
        self.chunk_size = chunk_size
        super().__init__(alpha=alpha, fit_prior=fit_prior,
                         class_prior=class_prior)

    def fit(self, X, y, sample_weight=None):
        if self.chunk_size <= 0:
            return super().fit(X, y, sample_weight=sample_weight)

        classes = numpy.unique(y)
        for i in range(0, X.shape[1], self.chunk_size):
            cX = X[i:i+self.chunk_size]
            cy = y[i:i+self.chunk_size]
            if sample_weight is None:
                csample_weight = None
            else:
                csample_weight = sample_weight[i:i+self.chunk_size]
            self.partial_fit(cX, cy, sample_weight=csample_weight,
                             classes=classes)

        return self


class MultioutputClassifier(BaseEstimator, ClassifierMixin):
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
