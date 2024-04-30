# ----------------------------------------------------------------------------
# Copyright (c) 2016-2023, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from dataclasses import dataclass, field
from functools import cached_property
from itertools import islice, repeat
from typing import Dict, List

from joblib import Parallel, delayed


@dataclass
class _TaxonNode:
    # The _TaxonNode is used to build a hierarchy from a list of sorted class
    # labels. It allows one to quickly find class label indices of taxonomy
    # labels that satisfy a given taxonomy hierarchy. For example, given the
    # 'k__Bacteria' taxon, the _TaxonNode.range property will yield all class
    # label indices where 'k__Bacteria' is a prefix.

    name: str
    offset_index: int
    children: Dict[str, "_TaxonNode"] = field(
        default_factory=dict,
        repr=False)

    @classmethod
    def create_tree(cls, classes: List[str], separator: str):
        if not all(a <= b for a, b in zip(classes, classes[1:])):
            raise Exception("classes must be in sorted order")
        root = cls("Unassigned", 0)
        for class_start_index, label in enumerate(classes):
            taxons = label.split(separator)
            node = root
            for name in taxons:
                if name not in node.children:
                    node.children[name] = cls(name, class_start_index)
                node = node.children[name]
        return root

    @property
    def range(self) -> range:
        return range(
            self.offset_index,
            self.offset_index + self.num_leaf_nodes)

    @cached_property
    def num_leaf_nodes(self) -> int:
        if len(self.children) == 0:
            return 1
        return sum(c.num_leaf_nodes for c in self.children.values())


_specific_fitters = [
        ['naive_bayes',
         [['feat_ext',
           {'__type__': 'feature_extraction.text.HashingVectorizer',
            'analyzer': 'char_wb',
            'n_features': 8192,
            'ngram_range': (7, 7),
            'alternate_sign': False}],
          ['classify',
           {'__type__': 'custom.LowMemoryMultinomialNB',
            'alpha': 0.001,
            'fit_prior': False}]]]]


def fit_pipeline(reads, taxonomy, pipeline):
    seq_ids, X = _extract_reads(reads)
    data = [(taxonomy[s], x) for s, x in zip(seq_ids, X) if s in taxonomy]
    y, X = list(zip(*data))
    pipeline.fit(X, y)
    return pipeline


def _extract_reads(reads):
    return zip(*[(r.metadata['id'], r._string) for r in reads])


def predict(reads, pipeline, separator=';', chunk_size=262144, n_jobs=1,
            pre_dispatch='2*n_jobs', confidence='disable'):
    jobs = (
        delayed(_predict_chunk)(pipeline, separator, confidence, chunk)
        for chunk in _chunks(reads, chunk_size))
    workers = Parallel(n_jobs=n_jobs, batch_size=1, pre_dispatch=pre_dispatch)
    for calculated in workers(jobs):
        yield from calculated


def _predict_chunk(pipeline, separator, confidence, chunk):
    if confidence == 'disable':
        return _predict_chunk_without_conf(pipeline, chunk)
    else:
        return _predict_chunk_with_conf(pipeline, separator, confidence, chunk)


def _predict_chunk_without_conf(pipeline, chunk):
    seq_ids, X = _extract_reads(chunk)
    y = pipeline.predict(X)
    return zip(seq_ids, y, repeat(-1.))


def _predict_chunk_with_conf(pipeline, separator, confidence, chunk):
    seq_ids, X = _extract_reads(chunk)

    if not hasattr(pipeline, "predict_proba"):
        raise ValueError('this classifier does not support confidence values')
    prob_pos = pipeline.predict_proba(X)
    if prob_pos.shape != (len(X), len(pipeline.classes_)):
        raise ValueError('this classifier does not support confidence values')

    y = pipeline.classes_[prob_pos.argmax(axis=1)]

    taxonomy_tree = _TaxonNode.create_tree(pipeline.classes_, separator)

    results = []
    for seq_id, taxon, class_probs in zip(seq_ids, y, prob_pos):
        split_taxon = taxon.split(separator)
        accepted_cum_prob = 0.0
        cum_prob = 0.0
        result = []
        current = taxonomy_tree
        for rank in split_taxon:
            current = current.children[rank]
            cum_prob = class_probs[current.range].sum()
            if cum_prob < confidence:
                break
            accepted_cum_prob = cum_prob
            result.append(rank)
        if len(result) == 0:
            results.append((seq_id, "Unassigned", 1.0 - cum_prob))
        else:
            results.append((seq_id, separator.join(result), accepted_cum_prob))
    return results


def _chunks(reads, chunk_size):
    reads = iter(reads)
    while True:
        chunk = list(islice(reads, chunk_size))
        if len(chunk) == 0:
            break
        yield chunk
