# ----------------------------------------------------------------------------
# Copyright (c) 2016--, QIIME development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import pickle
import json

import qiime.plugin
import qiime.plugin.model as model

from .plugin_setup import plugin


# Constants
_n_bytes = 2**31
_max_bytes = _n_bytes - 1


# Semantic Types
TaxonomicClassifier = qiime.plugin.SemanticType('TaxonomicClassifier')


# Formats
# https://github.com/qiime2/q2-types/issues/49
class PickleFormat(model.BinaryFileFormat):
    def sniff(self):
        # Trying to detect a large pickled file seems difficult to the point of
        # not being worth it. http://stackoverflow.com/q/13939913/313548
        return True


# https://github.com/qiime2/q2-types/issues/49
class JSONFormat(model.TextFileFormat):
    def sniff(self):
        with self.open() as fh:
            try:
                json.load(fh)
                return True
            except json.JSONDecodeError:
                pass
        return False


class TaxonomicClassifierDirFmt(model.DirectoryFormat):
    preprocess_params = model.File('preprocess_params.json', format=JSONFormat)
    sklearn_pipeline = model.File('sklearn_pipeline.pkl', format=PickleFormat)


# Transformers
@plugin.register_transformer
def _1(dirfmt: TaxonomicClassifierDirFmt) -> dict:
    # Note: the next two lines will likely disappear when cleanup of the
    # views/transformers API takes place.
    preprocess_params = dirfmt.preprocess_params.view(JSONFormat)
    sklearn_pipeline = dirfmt.sklearn_pipeline.view(PickleFormat)

    with preprocess_params.open() as fh:
        params = json.load(fh)
    # Macs can't pickle or unpickle objects larger than ~2GB. See
    # http://bugs.python.org/issue24658 Thanks for the workaround:
    # http://stackoverflow.com/q/31468117/313548
    input_size = sklearn_pipeline.path.stat().st_size
    bytes_in = bytearray(input_size)
    with sklearn_pipeline.open() as fh:
        for idx in range(0, input_size, _max_bytes):
            bytes_in[idx:idx+_max_bytes] = fh.read(_max_bytes)
    pipeline = pickle.loads(bytes_in)
    return {'params': params, 'pipeline': pipeline}


@plugin.register_transformer
def _2(data: dict) -> TaxonomicClassifierDirFmt:
    if 'params' not in data:
        raise ValueError('classifier does not contain params')
    if 'pipeline' not in data:
        raise ValueError('classifier does not contain pipeline')

    preprocess_params = JSONFormat()
    with preprocess_params.open() as fh:
        json.dump(data['params'], fh)

    sklearn_pipeline = PickleFormat()
    # See above comment about pickle on macs.
    bytes_out = pickle.dumps(data['pipeline'])
    with sklearn_pipeline.open() as fh:
        for idx in range(0, len(bytes_out), _max_bytes):
            fh.write(bytes_out[idx:idx+_max_bytes])

    dirfmt = TaxonomicClassifierDirFmt()
    dirfmt.preprocess_params.write_data(preprocess_params, JSONFormat)
    dirfmt.sklearn_pipeline.write_data(sklearn_pipeline, PickleFormat)

    return dirfmt

# Registrations
plugin.register_semantic_type(TaxonomicClassifier)
plugin.register_semantic_type_to_format(
    TaxonomicClassifier, artifact_format=TaxonomicClassifierDirFmt)
