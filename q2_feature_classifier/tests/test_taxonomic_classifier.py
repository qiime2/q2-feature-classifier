# ----------------------------------------------------------------------------
# Copyright (c) 2016--, QIIME development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import unittest
import tempfile
import tarfile
import os
import shutil

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from qiime2.sdk import Artifact
from qiime2.plugins.feature_classifier.methods import \
    fit_classifier_naive_bayes

from .._taxonomic_classifier import (
    TaxonomicClassifier, TaxonomicClassifierDirFmt, PickleFormat)
from . import FeatureClassifierTestPluginBase


class TaxonomicClassifierTestBase(FeatureClassifierTestPluginBase):
    package = 'q2_feature_classifier.tests'

    def setUp(self):
        super().setUp()

        reads = Artifact.import_data(
            'FeatureData[Sequence]',
            self.get_data_path('se-dna-sequences.fasta'))
        taxonomy = Artifact.import_data(
            'FeatureData[Taxonomy]', self.get_data_path('taxonomy.tsv'))
        classifier = fit_classifier_naive_bayes(reads, taxonomy)
        pipeline = classifier.classifier.view(Pipeline)
        transformer = self.get_transformer(Pipeline, TaxonomicClassifierDirFmt)
        self._sklp = transformer(pipeline)
        sklearn_pipeline = self._sklp.sklearn_pipeline.view(PickleFormat)
        self.sklearn_pipeline = str(sklearn_pipeline)


class TestTypes(FeatureClassifierTestPluginBase):
    def test_taxonomic_classifier_semantic_type_registration(self):
        self.assertRegisteredSemanticType(TaxonomicClassifier)

    def test_taxonomic_classifier_semantic_type_to_format_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            TaxonomicClassifier, TaxonomicClassifierDirFmt)


class TestFormats(TaxonomicClassifierTestBase):
    def test_taxonomic_classifier_dir_fmt(self):
        shutil.copy(self.sklearn_pipeline, self.temp_dir.name)
        format = TaxonomicClassifierDirFmt(self.temp_dir.name, mode='r')

        # Should not error
        format.validate()


class TestTransformers(TaxonomicClassifierTestBase):
    def test_taxo_class_dir_fmt_to_taxo_class_result(self):
        shutil.copy(self.sklearn_pipeline, self.temp_dir.name)
        input = TaxonomicClassifierDirFmt(self.temp_dir.name, mode='r')

        transformer = self.get_transformer(TaxonomicClassifierDirFmt, Pipeline)
        obs = transformer(input)

        self.assertTrue(obs)

    def test_taxo_class_result_to_taxo_class_dir_fmt(self):
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

        exp = read_pipeline(self.sklearn_pipeline)
        transformer = self.get_transformer(Pipeline, TaxonomicClassifierDirFmt)
        obs = transformer(exp)
        sklearn_pipeline = obs.sklearn_pipeline.view(PickleFormat)
        obs_pipeline = read_pipeline(str(sklearn_pipeline))
        obs = obs_pipeline
        self.assertTrue(obs)


if __name__ == "__main__":
    unittest.main()
