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

_n_bytes = 2**31
_max_bytes = _n_bytes - 1


def read_classifier(data_dir):
    with open(os.path.join(data_dir, 'preprocess_params.json'), 'r') as fh:
        params = json.load(fh)
    # Macs can't pickle or unpickle objects larger than ~2GB. See
    # http://bugs.python.org/issue24658
    # Thanks for the workaround
    # http://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb # noqa
    pickle_path = os.path.join(data_dir, 'sklearn_pipeline.pkl')
    input_size = os.path.getsize(pickle_path)
    bytes_in = bytearray(input_size)
    with open(pickle_path, 'rb') as fh:
        for idx in range(0, input_size, _max_bytes):
            bytes_in[idx:idx+_max_bytes] = fh.read(_max_bytes)
    pipeline = pickle.loads(bytes_in)

    return {'params': params, 'pipeline': pipeline}


def write_classifier(view, data_dir):
    if 'params' not in view:
        raise ValueError('classifier does not contain params')
    with open(os.path.join(data_dir, 'preprocess_params.json'), 'w') as fh:
        json.dump(view['params'], fh)

    if 'pipeline' not in view:
        raise ValueError('classifier does not contain pipeline')
    # See above comment about pickle on macs.
    bytes_out = pickle.dumps(view['pipeline'])
    with open(os.path.join(data_dir, 'sklearn_pipeline.pkl'), 'wb') as fh:
        for idx in range(0, len(bytes_out), _max_bytes):
            fh.write(bytes_out[idx:idx+_max_bytes])

plugin.register_data_layout_reader('taxonomic_classifier', 1, dict,
                                   read_classifier)
plugin.register_data_layout_writer('taxonomic_classifier', 1, dict,
                                   write_classifier)
