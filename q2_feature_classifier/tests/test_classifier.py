# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import json

from qiime.sdk import Artifact
from qiime.plugins import feature_classifier
import pandas as pd

from q2_feature_classifier._skl import _specific_fitters

from . import FeatureClassifierTestPluginBase


class ClassifierTests(FeatureClassifierTestPluginBase):
    package = 'q2_feature_classifier.tests'

    def setUp(self):
        super().setUp()
        self.taxonomy = Artifact.import_data(
            'FeatureData[Taxonomy]', self.get_data_path('taxonomy.tsv'))

    def test_fit_classifier_paired_end(self):
        # fit_classifier_paired_end should generate a taxonomic_classifier
        reads = Artifact.import_data('FeatureData[PairedEndSequence]',
                                     self.get_data_path('pe-dna-sequences'))

        classifier_specification = \
            {'steps': [['vectorize', 'feature_extraction.DictVectorizer'],
                       ['transform', 'feature_selection.SelectPercentile'],
                       ['classify', 'naive_bayes.MultinomialNB']]}
        classifier_specification = json.dumps(classifier_specification)
        fit_classifier = feature_classifier.methods.fit_classifier_paired_end
        result = fit_classifier(reads, self.taxonomy, classifier_specification,
                                taxonomy_depth=2, taxonomy_separator='; ')

        classify = feature_classifier.methods.classify_paired_end
        result = classify(reads, result.classifier)

        ref = self.taxonomy.view(pd.Series).to_dict()
        cls = result.classification.view(pd.Series).to_dict()

        right = 0.
        for taxon in cls:
            right += ref[taxon].startswith(cls[taxon])
        self.assertGreater(right/len(cls), 0.5)

    def test_fit_classifier(self):
        # fit_classifier should generate a working taxonomic_classifier
        reads = Artifact.import_data(
            'FeatureData[Sequence]',
            self.get_data_path('se-dna-sequences.fasta'))

        classifier_specification = \
            {'steps': [['vectorize', 'feature_extraction.DictVectorizer'],
                       ['transform', 'feature_selection.SelectPercentile'],
                       ['classify', 'naive_bayes.MultinomialNB']]}
        classifier_specification = json.dumps(classifier_specification)
        fit_classifier = feature_classifier.methods.fit_classifier
        result = fit_classifier(reads, self.taxonomy, classifier_specification,
                                taxonomy_depth=2, taxonomy_separator='; ')

        classify = feature_classifier.methods.classify
        result = classify(reads, result.classifier)

        ref = self.taxonomy.view(pd.Series).to_dict()
        cls = result.classification.view(pd.Series).to_dict()

        right = 0.
        for taxon in cls:
            right += ref[taxon].startswith(cls[taxon])
        self.assertGreater(right/len(cls), 0.5)

    def test_fit_specific_classifiers(self):
        # specific and general classifiers should produce the same results
        gen_fitter = feature_classifier.methods.fit_classifier
        classify = feature_classifier.methods.classify
        reads = Artifact.import_data(
            'FeatureData[Sequence]',
            self.get_data_path('se-dna-sequences.fasta'))

        for name, spec in _specific_fitters:
            classifier_spec = json.dumps(spec)
            result = gen_fitter(reads, self.taxonomy, classifier_spec,
                                taxonomy_depth=7, taxonomy_separator='; ')
            result = classify(reads, result.classifier)
            gc = result.classification.view(pd.Series).to_dict()
            spec_fitter = getattr(feature_classifier.methods,
                                  'fit_classifier_' + name)
            result = spec_fitter(reads, self.taxonomy, taxonomy_depth=7,
                                 taxonomy_separator='; ')
            result = classify(reads, result.classifier)
            sc = result.classification.view(pd.Series).to_dict()
            for taxon in gc:
                self.assertEqual(gc[taxon], sc[taxon])

    def test_fit_specific_classifiers_paired_end(self):
        # specific and general classifiers should produce the same results
        gen_fitter = feature_classifier.methods.fit_classifier_paired_end
        classify = feature_classifier.methods.classify_paired_end
        reads = Artifact.import_data('FeatureData[PairedEndSequence]',
                                     self.get_data_path('pe-dna-sequences'))

        for name, spec in _specific_fitters:
            classifier_spec = json.dumps(spec)
            result = gen_fitter(reads, self.taxonomy, classifier_spec,
                                taxonomy_depth=7, taxonomy_separator='; ')
            result = classify(reads, result.classifier)
            gc = result.classification.view(pd.Series).to_dict()
            spec_fitter = getattr(feature_classifier.methods,
                                  'fit_classifier_' + name + '_paired_end')
            result = spec_fitter(reads, self.taxonomy, taxonomy_depth=7,
                                 taxonomy_separator='; ')
            result = classify(reads, result.classifier)
            sc = result.classification.view(pd.Series).to_dict()
            for taxon in gc:
                self.assertEqual(gc[taxon], sc[taxon])
