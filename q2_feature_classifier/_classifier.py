# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import types

import pandas as pd

from ._skl import train_assigner_sklearn
from ._perfect import train_assigner_perfect

def classify(sequences : types.GeneratorType, reference_taxonomy : pd.Series,
        reference_sequences : types.GeneratorType, method : str,
        depth : int = 4) -> pd.Series:
    id_to_taxon = {}
    for _id, taxon in reference_taxonomy.to_dict().items():
        id_to_taxon[_id] = '; '.join(taxon.split('; ')[:depth])
    reference = ((s,) for s in reference_sequences)
    assign = train_assigner(reference, id_to_taxon, method=method)
    classification = {s.metadata['id'] :
                      '; '.join(assign((s,))) for s in sequences}
    result = pd.Series(classification)
    result.name = 'taxonomy'
    result.index.name = 'Feature ID'
    return result

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
