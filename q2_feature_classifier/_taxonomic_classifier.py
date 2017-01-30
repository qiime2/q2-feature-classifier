# ----------------------------------------------------------------------------
# Copyright (c) 2016-2017, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import json
import tarfile
import os

import sklearn
from sklearn.externals import joblib
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
    preprocess_params = model.File('preprocess_params.json', format=JSONFormat)
    sklearn_pipeline = model.File('sklearn_pipeline.tar', format=PickleFormat)


class VersionedTaxonomicClassifierDirFmt(TaxonomicClassifierDirFmt):
    version_info = model.File('sklearn_version.json', format=JSONFormat)


# Transformers
@plugin.register_transformer
def _1(dirfmt: VersionedTaxonomicClassifierDirFmt) -> dict:
    sklearn_version = dirfmt.version_info.view(dict)['sklearn-version']
    if sklearn_version != sklearn.__version__:
        raise ValueError('The scikit-learn version (%s) used to generate this'
                         ' artifact does not match the current version'
                         ' of scikit-learn installed (%s). Please retrain your'
                         ' classifier for your current deployment to prevent'
                         ' data-corruption errors.'
                         % (sklearn_version, sklearn.__version__))

    sklearn_pipeline = dirfmt.sklearn_pipeline.view(PickleFormat)
    params = dirfmt.preprocess_params.view(dict)

    with tarfile.open(str(sklearn_pipeline)) as tar:
        dirname = os.path.dirname(str(sklearn_pipeline))
        tar.extractall(dirname)
        pipeline = joblib.load(os.path.join(dirname, 'sklearn_pipeline.pkl'))
        for fn in tar.getnames():
            os.unlink(os.path.join(dirname, fn))

    return {'params': params, 'pipeline': pipeline}


@plugin.register_transformer
def _2(data: dict) -> VersionedTaxonomicClassifierDirFmt:
    if 'params' not in data:
        raise ValueError('classifier does not contain params')
    if 'pipeline' not in data:
        raise ValueError('classifier does not contain pipeline')

    sklearn_pipeline = PickleFormat()
    with tarfile.open(str(sklearn_pipeline), 'w') as tar:
        pf = os.path.join(os.path.dirname(str(sklearn_pipeline)),
                          'sklearn_pipeline.pkl')
        for fn in joblib.dump(data['pipeline'], pf):
            tar.add(fn, os.path.basename(fn))
            os.unlink(fn)

    dirfmt = VersionedTaxonomicClassifierDirFmt()
    dirfmt.version_info.write_data(
        {'sklearn-version': sklearn.__version__}, dict)
    dirfmt.preprocess_params.write_data(data['params'], dict)
    dirfmt.sklearn_pipeline.write_data(sklearn_pipeline, PickleFormat)

    return dirfmt


@plugin.register_transformer
def _3(dirfmt: TaxonomicClassifierDirFmt) -> dict:
    raise ValueError('The scikit-learn version could not be determined for'
                     ' this artifact, please retrain your classifier for your'
                     ' current deployment to prevent data-corruption errors.')


@plugin.register_transformer
def _4(fmt: JSONFormat) -> dict:
    with fmt.open() as fh:
        return json.load(fh)


@plugin.register_transformer
def _5(data: dict) -> JSONFormat:
    result = JSONFormat()
    with result.open() as fh:
        json.dump(data, fh)
    return result


# Registrations
plugin.register_semantic_types(TaxonomicClassifier)
plugin.register_formats(PickleFormat, JSONFormat, TaxonomicClassifierDirFmt,
                        VersionedTaxonomicClassifierDirFmt)
plugin.register_semantic_type_to_format(
    TaxonomicClassifier, artifact_format=VersionedTaxonomicClassifierDirFmt)
