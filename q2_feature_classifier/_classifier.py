# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import types

import pandas as pd
from qiime.plugin import Int, Str, Choices
from q2_types import (ReferenceFeatures, SSU, FeatureData, Taxonomy, Sequence,
                      PairedEndSequence)

from ._skl import train_assigner_sklearn
from ._perfect import train_assigner_perfect
from .plugin_setup import plugin


def classify(sequences: types.GeneratorType, reference_taxonomy: pd.Series,
             reference_sequences: types.GeneratorType, method: str
             ) -> pd.Series:
    reference = ((s,) for s in reference_sequences)
    assign = train_assigner(reference, reference_taxonomy, method=method)
    classification = {s.metadata['id']: assign((s,)) for s in sequences}
    result = pd.Series(classification)
    result.name = 'taxonomy'
    result.index.name = 'Feature ID'
    return result

plugin.methods.register_function(
    function=classify,
    inputs={'sequences': FeatureData[Sequence],
            'reference_sequences': ReferenceFeatures[SSU],
            'reference_taxonomy': ReferenceFeatures[SSU]},
    parameters={'method': Str % Choices(['naive-bayes', 'svc', 'perfect'])},
    outputs=[('classification', FeatureData[Taxonomy])],
    name='Train and apply feature classifier.',
    description='Train a classifier and apply it to feature data.'
)


def classify_paired_end(pairs: types.GeneratorType,
                        reference_taxonomy: pd.Series,
                        reference_sequences: types.GeneratorType, method: str
                        ) -> pd.Series:
    assign = train_assigner(reference_sequences, reference_taxonomy,
                            method=method)
    classification = {s[0].metadata['id']: assign(s) for s in pairs}
    result = pd.Series(classification)
    result.name = 'taxonomy'
    result.index.name = 'Feature ID'
    return result

plugin.methods.register_function(
    function=classify_paired_end,
    inputs={'pairs': FeatureData[PairedEndSequence],
            'reference_sequences': FeatureData[PairedEndSequence],
            'reference_taxonomy': ReferenceFeatures[SSU]},
    parameters={'method': Str % Choices(['naive-bayes', 'svc', 'perfect'])},
    outputs=[('classification', FeatureData[Taxonomy])],
    name='Train and apply feature classifier for paired-end data.',
    description='Train a classifier and apply it to paired-end feature data.'
)


def train_assigner(reads, taxonomy, method='naive-bayes'):
    """ Create a function that assigns a taxonomy to a read or reads.

    Parameters
    ----------
    reads : list
        list of single or pairs of skbio.sequence.DNA reads
    taxonomy : dict
        mapping from taxon id to taxonomic classification
    method : str, optional
        method to use for assignment. 'perfect' uses an inverse dict for
        perfect recall. Can return multiple classifications per read. 'SVM' to
        use sklearn.svm.SVC. 'NB' to use sklearn.naive_bays.MultinomialNB.

    Returns
    -------
    callable
        function that takes a read or read pair and returns a list of
        classifications
    """

    if method == 'naive-bayes':
        return train_assigner_sklearn(reads, taxonomy, 'NB')
    elif method == 'svc':
        return train_assigner_sklearn(reads, taxonomy, 'SVM')
    elif method == 'perfect':
        return train_assigner_perfect(reads, taxonomy)
    else:
        raise NotImplementedError(method + ' method not supported')
