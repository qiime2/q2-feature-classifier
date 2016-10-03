# ----------------------------------------------------------------------------
# Copyright (c) 2016--, QIIME development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import json
import tarfile
import os

from sklearn.externals import joblib
import qiime.plugin
import qiime.plugin.model as model

from .plugin_setup import plugin


# Semantic Types
TaxonomicClassifier = qiime.plugin.SemanticType('TaxonomicClassifier')


# Formats
class PickleFormat(model.BinaryFileFormat):
    def sniff(self):
        return tarfile.is_tarfile(str(self))


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
    sklearn_pipeline = model.File('sklearn_pipeline.tar', format=PickleFormat)


# Transformers
@plugin.register_transformer
def _1(dirfmt: TaxonomicClassifierDirFmt) -> dict:
    # Note: the next two lines will likely disappear when cleanup of the
    # views/transformers API takes place.
    preprocess_params = dirfmt.preprocess_params.view(JSONFormat)
    sklearn_pipeline = dirfmt.sklearn_pipeline.view(PickleFormat)

    with preprocess_params.open() as fh:
        params = json.load(fh)

    with tarfile.open(str(sklearn_pipeline)) as tar:
        dirname = os.path.dirname(str(sklearn_pipeline))
        tar.extractall(dirname)
        pipeline = joblib.load(os.path.join(dirname, 'sklearn_pipeline.pkl'))
        for fn in tar.getnames():
            os.unlink(os.path.join(dirname, fn))

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
    with tarfile.open(str(sklearn_pipeline), 'w') as tar:
        pf = os.path.join(os.path.dirname(str(sklearn_pipeline)),
                          'sklearn_pipeline.pkl')
        for fn in joblib.dump(data['pipeline'], pf):
            tar.add(fn, os.path.basename(fn))
            os.unlink(fn)

    dirfmt = TaxonomicClassifierDirFmt()
    dirfmt.preprocess_params.write_data(preprocess_params, JSONFormat)
    dirfmt.sklearn_pipeline.write_data(sklearn_pipeline, PickleFormat)

    return dirfmt

# Registrations
plugin.register_semantic_types(TaxonomicClassifier)
plugin.register_formats(PickleFormat, JSONFormat, TaxonomicClassifierDirFmt)
plugin.register_semantic_type_to_format(
    TaxonomicClassifier, artifact_format=TaxonomicClassifierDirFmt)
