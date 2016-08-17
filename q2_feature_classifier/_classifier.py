# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import types
import json
import importlib

import pandas as pd
from qiime.plugin import Int, Str, Float
from q2_types import (ReferenceFeatures, SSU, FeatureData, Taxonomy, Sequence,
                      PairedEndSequence)
from sklearn.pipeline import Pipeline

from ._skl import fit_pipeline, predict, _specific_fitters
from ._taxonomic_classifier import TaxonomicClassifier
from .plugin_setup import plugin


def as_pipeline_params(json):
    if '__sklearn_class__' in json:
        module = importlib.import_module(json['module'])
        return getattr(module, json['class'])()
    return json


def fit_classifier(reference_reads: types.GeneratorType,
                   reference_taxonomy: pd.Series,
                   classifier_specification: str, word_length: int=8
                   ) -> dict:
    spec = json.loads(classifier_specification, object_hook=as_pipeline_params)
    params = {'word_length': word_length}
    pipeline = fit_pipeline(reference_reads, reference_taxonomy,
                            spec, **params)
    return {'params': params, 'pipeline': pipeline}

plugin.methods.register_function(
    function=fit_classifier,
    inputs={'reference_reads': FeatureData[Sequence],
            'reference_taxonomy': ReferenceFeatures[SSU]},
    parameters={'classifier_specification': Str, 'word_length': Int},
    outputs=[('taxonomic_classifier', TaxonomicClassifier)],
    name='Train a scikit-learn classifier.',
    description='Train a scikit-learn classifier to classify reads.'
)

fit_classifier.__name__ = 'fit_classifier_paired_end'
plugin.methods.register_function(
    function=fit_classifier,
    inputs={'reference_reads': FeatureData[PairedEndSequence],
            'reference_taxonomy': ReferenceFeatures[SSU]},
    parameters={'classifier_specification': Str, 'word_length': Int},
    outputs=[('classifier', TaxonomicClassifier)],
    name='Train a scikit-learn classifier.',
    description='Train a scikit-learn classifier to classify paired end reads.'
)


def classify(reads: types.GeneratorType, classifier: dict) -> pd.Series:
    seq_ids, classifications = predict(reads, classifier['pipeline'],
                                       **classifier['params'])
    result = pd.Series(classifications, index=seq_ids)
    result.name = 'taxonomy'
    result.index.name = 'Feature ID'
    return result

plugin.methods.register_function(
    function=classify,
    inputs={'reads': FeatureData[Sequence],
            'classifier': TaxonomicClassifier},
    parameters={},
    outputs=[('classification', FeatureData[Taxonomy])],
    name='Classify reads by taxon.',
    description='Classify reads by taxon using a fitted classifier.',
)

classify.__name__ = 'classify_paired_end'
plugin.methods.register_function(
    function=classify,
    inputs={'reads': FeatureData[PairedEndSequence],
            'classifier': TaxonomicClassifier},
    parameters={},
    outputs=[('classification', FeatureData[Taxonomy])],
    name='Classify reads by taxon.',
    description='Classify reads by taxon using a fitted classifier.',
)

class PipelineParamEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            if hasattr(obj, 'tolist'):
                return self.default(obj.tolist())
            else:
                print(obj)
                return {'module': obj.__module__,
                        'class': obj.__class__.__name__,
                        '__sklearn_class__': True}

def _register_fitter(name, steps):
    pipeline = Pipeline(steps)
    prefix = steps[-1][0] + '__'
    annotations = {}
    parameters = {}

    type_map = {int: Int, float: Float}  # EEE add bool when it arrives
    spec = pipeline.get_params()
    for param, value in spec.items():
        if not param.startswith(prefix) or callable(value):  # ignore functions
            continue
        param = param[len(prefix):]
        parameters[param] = type_map.get(type(value), Str)
        annotations[param] = type(value) if type(value) in type_map else str

    def _generic_fitter(reference_reads: types.GeneratorType,
                        reference_taxonomy: pd.Series,
                        word_length: int=8, **kwargs) -> dict:
        for param, value in kwargs.items():  # EEE needs work
            try:
                spec[prefix + param] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                spec[prefix + param] = value
        flat_spec = json.dumps(spec, cls=PipelineParamEncoder)
        return fit_classifier(reference_reads, reference_taxonomy, flat_spec,
                              word_length=word_length)

    parameters.update({'word_length': Int})
    _generic_fitter.__annotations__.update(annotations)
    _generic_fitter.__name__ = 'fit_classifier_' + name
    plugin.methods.register_function(
        function=_generic_fitter,
        inputs={'reference_reads': FeatureData[Sequence],
                'reference_taxonomy': ReferenceFeatures[SSU]},
        parameters=parameters,
        outputs=[('classifier', TaxonomicClassifier)],
        name='Train the scikit-learn ' + \
             spec['steps'][-1][1].__class__.__name__ + \
             ' classifier.',
        description='Create a ' + \
                    spec['steps'][-1][1].__class__.__name__ + \
                    ' classifier for reads'
    )
    _generic_fitter.__name__ = 'fit_classifier_' + name + '_paired_end'
    plugin.methods.register_function(
        function=_generic_fitter,
        inputs={'reference_reads': FeatureData[PairedEndSequence],
                'reference_taxonomy': ReferenceFeatures[SSU]},
        parameters=parameters,
        outputs=[('classifier', TaxonomicClassifier)],
        name='Train the scikit-learn ' + \
             spec['steps'][-1][1].__class__.__name__ + \
             ' classifier.',
        description='Create a ' + \
                    spec['steps'][-1][1].__class__.__name__ + \
                    ' classifier for paired-end reads'
    )

for name, pipeline in _specific_fitters:
    _register_fitter(name, pipeline)
