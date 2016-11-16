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
from sklearn.pipeline import Pipeline
import qiime2.plugin
import qiime2.plugin.model as model

from .plugin_setup import plugin


# Semantic Types
TaxonomicClassifier = qiime2.plugin.SemanticType('TaxonomicClassifier')


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
    sklearn_pipeline = model.File('sklearn_pipeline.tar', format=PickleFormat)


# Transformers
@plugin.register_transformer
def _1(dirfmt: TaxonomicClassifierDirFmt) -> Pipeline:
    # Note: the next two lines will likely disappear when cleanup of the
    # views/transformers API takes place.
    sklearn_pipeline = dirfmt.sklearn_pipeline.view(PickleFormat)

    with tarfile.open(str(sklearn_pipeline)) as tar:
        tmpdir = model.DirectoryFormat()
        dirname = str(tmpdir)
        tar.extractall(dirname)
        pipeline = joblib.load(os.path.join(dirname, 'sklearn_pipeline.pkl'))
        for fn in tar.getnames():
            os.unlink(os.path.join(dirname, fn))

    return pipeline


@plugin.register_transformer
def _2(data: Pipeline) -> TaxonomicClassifierDirFmt:
    sklearn_pipeline = PickleFormat()
    with tarfile.open(str(sklearn_pipeline), 'w') as tar:
        tmpdir = model.DirectoryFormat()
        pf = os.path.join(str(tmpdir), 'sklearn_pipeline.pkl')
        for fn in joblib.dump(data, pf):
            tar.add(fn, os.path.basename(fn))
            os.unlink(fn)

    dirfmt = TaxonomicClassifierDirFmt()
    dirfmt.sklearn_pipeline.write_data(sklearn_pipeline, PickleFormat)

    return dirfmt

# Registrations
plugin.register_semantic_types(TaxonomicClassifier)
plugin.register_formats(PickleFormat, JSONFormat, TaxonomicClassifierDirFmt)
plugin.register_semantic_type_to_format(
    TaxonomicClassifier, artifact_format=TaxonomicClassifierDirFmt)
