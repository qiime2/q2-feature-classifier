# ----------------------------------------------------------------------------
# Copyright (c) 2016--, QIIME development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import os.path
import pickle
import json

import qiime.plugin

from .plugin_setup import plugin

TaxonomicClassifier = qiime.plugin.SemanticType('TaxonomicClassifier')

plugin.register_semantic_type(TaxonomicClassifier)


class PickleFormat(qiime.plugin.FileFormat):
    name = 'pickle'

    @classmethod
    def sniff(cls, filepath):
        # Trying to detect a large pickled file seems difficult to the
        # point of not being worth it.
        # http://stackoverflow.com/questions/13939913/how-to-test-if-a-file-has-been-created-by-pickle # noqa
        return True


class JSONFormat(qiime.plugin.FileFormat):
    name = 'json'

    @classmethod
    def sniff(cls, filepath):
        with open(filepath, 'r') as fh:
            try:
                json.load(fh)
                return True
            except json.JSONDecodeError:
                pass
        return False


taxonomic_classifier_data_layout = \
        qiime.plugin.DataLayout('taxonomic_classifier', 1)
taxonomic_classifier_data_layout.register_file('sklearn_pipeline.pkl',
                                               PickleFormat)
taxonomic_classifier_data_layout.register_file('preprocess_params.json',
                                               JSONFormat)

plugin.register_data_layout(taxonomic_classifier_data_layout)
plugin.register_type_to_data_layout(TaxonomicClassifier,
                                    'taxonomic_classifier', 1)


def read_classifier(data_dir):
    with open(os.path.join(data_dir, 'preprocess_params.json'), 'r') as fh:
        params = json.load(fh)
    with open(os.path.join(data_dir, 'sklearn_pipeline.pkl'), 'rb') as fh:
        pipeline = pickle.load(fh)
    return {'params': params, 'pipeline': pipeline}


def write_classifier(view, data_dir):
    if 'params' not in view:
        raise ValueError('classifier does not contain params')
    with open(os.path.join(data_dir, 'preprocess_params.json'), 'w') as fh:
        json.dump(view['params'], fh)

    if 'pipeline' not in view:
        raise ValueError('classifier does not contain pipeline')
    with open(os.path.join(data_dir, 'sklearn_pipeline.pkl'), 'wb') as fh:
        pickle.dump(view['pipeline'], fh)

plugin.register_data_layout_reader('taxonomic_classifier', 1, dict,
                                   read_classifier)
plugin.register_data_layout_writer('taxonomic_classifier', 1, dict,
                                   write_classifier)
