# ----------------------------------------------------------------------------
# Copyright (c) 2016--, QIIME development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import unittest
import json
import tempfile
import tarfile
import os

from sklearn.externals import joblib

from .._taxonomic_classifier import (
    TaxonomicClassifier, TaxonomicClassifierDirFmt, JSONFormat, PickleFormat)
from . import FeatureClassifierTestPluginBase


class TestTypes(FeatureClassifierTestPluginBase):
    def test_taxonomic_classifier_semantic_type_registration(self):
        self.assertRegisteredSemanticType(TaxonomicClassifier)

    def test_taxonomic_classifier_semantic_type_to_format_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            TaxonomicClassifier, TaxonomicClassifierDirFmt)


class TestFormats(FeatureClassifierTestPluginBase):
    package = 'q2_feature_classifier.tests'

    def test_taxonomic_classifier_dir_fmt(self):
        format = self._setup_dir(['sklearn_pipeline.tar',
                                  'preprocess_params.json'],
                                 TaxonomicClassifierDirFmt)
        # Should not error
        format.validate()


class TestTransformers(FeatureClassifierTestPluginBase):
    package = 'q2_feature_classifier.tests'

    def test_taxo_class_dir_fmt_to_taxo_class_result(self):
        transformer = self.get_transformer(TaxonomicClassifierDirFmt, dict)
        input = self._setup_dir(['sklearn_pipeline.tar',
                                 'preprocess_params.json'],
                                TaxonomicClassifierDirFmt)

        obs = transformer(input)
        exp = ['params', 'pipeline']

        self.assertSetEqual(set(obs.keys()), set(exp))

    def test_taxo_class_result_to_taxo_class_dir_fmt(self):
        transformer = self.get_transformer(dict, TaxonomicClassifierDirFmt)
        params_filepath = self.get_data_path('preprocess_params.json')
        pipeline_filepath = self.get_data_path('sklearn_pipeline.tar')

        with open(params_filepath) as fh:
            params = json.load(fh)

        def read_pipeline(pipeline_filepath):
            with tarfile.open(pipeline_filepath) as tar:
                dirname = tempfile.mkdtemp()
                tar.extractall(dirname)
                pipeline = joblib.load(os.path.join(dirname,
                                       'sklearn_pipeline.pkl'))
                for fn in tar.getnames():
                    os.unlink(os.path.join(dirname, fn))
                os.rmdir(dirname)
            return pipeline
        pipeline = read_pipeline(pipeline_filepath)

        exp = {'params': params, 'pipeline': pipeline}

        obs = transformer(exp)

        preprocess_params = obs.preprocess_params.view(JSONFormat)
        sklearn_pipeline = obs.sklearn_pipeline.view(PickleFormat)

        with preprocess_params.open() as fh:
            obs_params = json.load(fh)
        obs_pipeline = read_pipeline(str(sklearn_pipeline))

        obs = {'params': obs_params, 'pipeline': obs_pipeline}

        self.assertEqual(obs['params'], exp['params'])
        self.assertTrue(obs['pipeline'])


if __name__ == "__main__":
    unittest.main()
