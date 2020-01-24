# ----------------------------------------------------------------------------
# Copyright (c) 2016-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import json

from qiime2.sdk import Artifact
from qiime2.plugins import feature_classifier
from q2_types.feature_data import DNAIterator
from q2_feature_classifier.custom import ChunkedHashingVectorizer
from q2_feature_classifier._skl import _extract_reads
from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd

from . import FeatureClassifierTestPluginBase


class CustomTests(FeatureClassifierTestPluginBase):
    package = 'q2_feature_classifier.tests'

    def setUp(self):
        super().setUp()
        self.taxonomy = Artifact.import_data(
            'FeatureData[Taxonomy]', self.get_data_path('taxonomy.tsv'))

    def test_low_memory_multinomial_nb(self):
        # results should not depend on chunk size
        fitter = feature_classifier.methods.fit_classifier_sklearn
        classify = feature_classifier.methods.classify_sklearn
        reads = Artifact.import_data(
            'FeatureData[Sequence]',
            self.get_data_path('se-dna-sequences.fasta'))

        spec = [['feat_ext',
                {'__type__': 'feature_extraction.text.HashingVectorizer',
                 'analyzer': 'char',
                 'n_features': 8192,
                 'ngram_range': [8, 8],
                 'alternate_sign': False}],
                ['classify',
                 {'__type__': 'custom.LowMemoryMultinomialNB',
                  'alpha': 0.01,
                  'chunk_size': 20000}]]

        classifier_spec = json.dumps(spec)
        result = fitter(reads, self.taxonomy, classifier_spec)
        result = classify(reads, result.classifier)
        gc = result.classification.view(pd.Series).to_dict()

        spec[1][1]['chunk_size'] = 20
        classifier_spec = json.dumps(spec)
        result = fitter(reads, self.taxonomy, classifier_spec)
        result = classify(reads, result.classifier)
        sc = result.classification.view(pd.Series).to_dict()

        for taxon in gc:
            self.assertEqual(gc[taxon], sc[taxon])

    def test_chunked_hashing_vectorizer(self):
        # results should not depend on chunk size
        _, X = _extract_reads(Artifact.import_data(
            'FeatureData[Sequence]',
            self.get_data_path('se-dna-sequences.fasta')).view(DNAIterator))

        params = {'analyzer': 'char',
                  'n_features': 8192,
                  'ngram_range': [8, 8],
                  'alternate_sign': False}
        hv = HashingVectorizer(**params)
        unchunked = hv.fit_transform(X)

        for chunk_size in (-1, 3, 13):
            chv = ChunkedHashingVectorizer(chunk_size=chunk_size, **params)
            chunked = chv.fit_transform(X)
            for x1, x2 in zip(chunked, unchunked):
                self.assertTrue((x1.todense() == x2.todense()).all())
