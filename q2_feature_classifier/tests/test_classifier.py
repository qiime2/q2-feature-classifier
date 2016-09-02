# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import unittest
import os
import json
from os.path import join
import inspect

from qiime.sdk import Artifact
from qiime.plugins import feature_classifier
import pandas as pd

from q2_feature_classifier._skl import _specific_fitters
from q2_feature_classifier._classifier import _load_class


class ClassifierTests(unittest.TestCase):
    data_dir = join(os.path.dirname(os.path.abspath(__file__)), 'data')

    def test_fit_classifier_paired_end(self):
        # fit_classifier_paired_end should generate a taxonomic_classifier
        reads = Artifact.load(join(self.data_dir, '85_ref_pe_reads.qza'))
        taxonomy = Artifact.load(join(self.data_dir, '85_ref_feat.qza'))
        classifier_specification = \
            {'steps': [['transform',
                        'sklearn.feature_selection.SelectPercentile'],
                       ['classify',
                        'sklearn.naive_bayes.MultinomialNB']]}
        classifier_specification = json.dumps(classifier_specification)
        fit_classifier = feature_classifier.methods.fit_classifier_paired_end
        classifier = fit_classifier(reads, taxonomy, classifier_specification,
                                    taxonomy_depth=2, taxonomy_separator='; ')

        classify = feature_classifier.methods.classify_paired_end
        classification = classify(reads, classifier)

        ref = taxonomy.view(pd.Series).to_dict()
        cls = classification.view(pd.Series).to_dict()

        right = 0.
        for taxon in cls:
            right += ref[taxon].startswith(cls[taxon])
        self.assertGreater(right/len(cls), 0.5)

    def test_fit_classifier(self):
        # fit_classifier should generate a working taxonomic_classifier
        reads = Artifact.load(join(self.data_dir, '85_ref_se_reads.qza'))
        taxonomy = Artifact.load(join(self.data_dir, '85_ref_feat.qza'))
        classifier_specification = \
            {'steps': [['transform',
                        'sklearn.feature_selection.SelectPercentile'],
                       ['classify',
                        'sklearn.naive_bayes.MultinomialNB']]}
        classifier_specification = json.dumps(classifier_specification)
        fit_classifier = feature_classifier.methods.fit_classifier
        classifier = fit_classifier(reads, taxonomy, classifier_specification,
                                    taxonomy_depth=2, taxonomy_separator='; ')

        classify = feature_classifier.methods.classify
        classification = classify(reads, classifier)

        ref = taxonomy.view(pd.Series).to_dict()
        cls = classification.view(pd.Series).to_dict()

        right = 0.
        for taxon in cls:
            right += ref[taxon].startswith(cls[taxon])
        self.assertGreater(right/len(cls), 0.5)

        pass

    def test_fit_specific_classifiers(self):
        # specific and general classifiers should produce the same results
        gen_fitter = feature_classifier.methods.fit_classifier
        classify = feature_classifier.methods.classify
        for name, spec in _specific_fitters:
            reads = Artifact.load(join(self.data_dir, '85_ref_se_reads.qza'))
            taxonomy = Artifact.load(join(self.data_dir, '85_ref_feat.qza'))
            classifier_spec = json.dumps(spec)
            gen_classifier = gen_fitter(reads, taxonomy, classifier_spec,
                                        taxonomy_depth=7,
                                        taxonomy_separator='; ')
            gen_classification = classify(reads, gen_classifier)
            gc = gen_classification.view(pd.Series).to_dict()
            spec_fitter = getattr(feature_classifier.methods,
                                  'fit_classifier_' + name)
            class_name = spec['steps'][-1][1]
            params = spec.get(spec['steps'][-1][0], {})
            skl_classifier = _load_class(class_name)(**params)
            params = skl_classifier.get_params()
            signature = inspect.signature(_load_class(class_name))
            for param_name, param in signature.parameters.items():
                if callable(param.default):
                    del params[param_name]
                if type(param.default) not in {int, float, bool}:
                    params[param_name] = json.dumps(params[param_name])
            spec_classifier = spec_fitter(reads, taxonomy, taxonomy_depth=7,
                                          taxonomy_separator='; ',
                                          **params)
            spec_classification = classify(reads, spec_classifier)
            sc = spec_classification.view(pd.Series).to_dict()
            for taxon in gc:
                self.assertEqual(gc[taxon], sc[taxon])

    def test_fit_specific_classifiers_paired_end(self):
        # specific and general classifiers should produce the same results
        gen_fitter = feature_classifier.methods.fit_classifier_paired_end
        classify = feature_classifier.methods.classify_paired_end
        for name, spec in _specific_fitters:
            reads = Artifact.load(join(self.data_dir, '85_ref_pe_reads.qza'))
            taxonomy = Artifact.load(join(self.data_dir, '85_ref_feat.qza'))
            classifier_spec = json.dumps(spec)
            gen_classifier = gen_fitter(reads, taxonomy, classifier_spec,
                                        taxonomy_depth=7,
                                        taxonomy_separator='; ')
            gen_classification = classify(reads, gen_classifier)
            gc = gen_classification.view(pd.Series).to_dict()
            spec_fitter = getattr(feature_classifier.methods,
                                  'fit_classifier_' + name + '_paired_end')
            class_name = spec['steps'][-1][1]
            params = spec.get(spec['steps'][-1][0], {})
            skl_classifier = _load_class(class_name)(**params)
            params = skl_classifier.get_params()
            signature = inspect.signature(_load_class(class_name))
            for param_name, param in signature.parameters.items():
                if callable(param.default):
                    del params[param_name]
                if type(param.default) not in {int, float, bool}:
                    params[param_name] = json.dumps(params[param_name])
            spec_classifier = spec_fitter(reads, taxonomy, taxonomy_depth=7,
                                          taxonomy_separator='; ',
                                          **params)
            spec_classification = classify(reads, spec_classifier)
            sc = spec_classification.view(pd.Series).to_dict()
            for taxon in gc:
                self.assertEqual(gc[taxon], sc[taxon])
