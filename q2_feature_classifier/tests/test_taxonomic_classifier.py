# ----------------------------------------------------------------------------
# Copyright (c) 2016-2020, QIIME 2 development team.
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
import shutil

import sklearn
import joblib
from sklearn.pipeline import Pipeline
from qiime2.sdk import Artifact
from qiime2.plugins.feature_classifier.methods import \
    fit_classifier_naive_bayes

from .._taxonomic_classifier import (
    TaxonomicClassifierDirFmt, TaxonomicClassifier,
    TaxonomicClassiferTemporaryPickleDirFmt, PickleFormat)
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
        transformer = self.get_transformer(
            Pipeline, TaxonomicClassiferTemporaryPickleDirFmt)
        self._sklp = transformer(pipeline)
        sklearn_pipeline = self._sklp.sklearn_pipeline.view(PickleFormat)
        self.sklearn_pipeline = str(sklearn_pipeline)

    def _custom_setup(self, version):
        with open(os.path.join(self.temp_dir.name,
                               'sklearn_version.json'), 'w') as fh:
            fh.write(json.dumps({'sklearn-version': version}))
        shutil.copy(self.sklearn_pipeline, self.temp_dir.name)
        return TaxonomicClassiferTemporaryPickleDirFmt(
            self.temp_dir.name, mode='r')


class TestTypes(FeatureClassifierTestPluginBase):
    def test_taxonomic_classifier_semantic_type_registration(self):
        self.assertRegisteredSemanticType(TaxonomicClassifier)

    def test_taxonomic_classifier_semantic_type_to_format_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            TaxonomicClassifier, TaxonomicClassiferTemporaryPickleDirFmt)


class TestFormats(TaxonomicClassifierTestBase):
    def test_taxonomic_classifier_dir_fmt(self):
        format = self._custom_setup(sklearn.__version__)

        # Should not error
        format.validate()


class TestTransformers(TaxonomicClassifierTestBase):
    def test_old_sklearn_version(self):
        transformer = self.get_transformer(
            TaxonomicClassiferTemporaryPickleDirFmt, Pipeline)
        input = self._custom_setup('a very old version')
        with self.assertRaises(ValueError):
            transformer(input)

    def test_old_dirfmt(self):
        transformer = self.get_transformer(TaxonomicClassifierDirFmt, Pipeline)
        with open(os.path.join(self.temp_dir.name,
                               'preprocess_params.json'), 'w') as fh:
            fh.write(json.dumps([]))
        shutil.copy(self.sklearn_pipeline, self.temp_dir.name)
        input = TaxonomicClassifierDirFmt(
            self.temp_dir.name, mode='r')
        with self.assertRaises(ValueError):
            transformer(input)

    def test_taxo_class_dir_fmt_to_taxo_class_result(self):
        input = self._custom_setup(sklearn.__version__)

        transformer = self.get_transformer(
            TaxonomicClassiferTemporaryPickleDirFmt, Pipeline)
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
        transformer = self.get_transformer(
            Pipeline, TaxonomicClassiferTemporaryPickleDirFmt)
        obs = transformer(exp)
        sklearn_pipeline = obs.sklearn_pipeline.view(PickleFormat)
        obs_pipeline = read_pipeline(str(sklearn_pipeline))
        obs = obs_pipeline
        self.assertTrue(obs)


if __name__ == "__main__":
    unittest.main()
