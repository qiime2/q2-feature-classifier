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
import inspect
import copy

import pandas as pd
from qiime.plugin import Int, Str, Float
from q2_types import (ReferenceFeatures, SSU, FeatureData, Taxonomy, Sequence,
                      PairedEndSequence)
from sklearn.pipeline import Pipeline

from ._skl import fit_pipeline, predict, _specific_fitters
from ._taxonomic_classifier import TaxonomicClassifier
from .plugin_setup import plugin


def _load_class(classname):
    module, klass = classname.rsplit('.', 1)
    module = importlib.import_module(module)
    return getattr(module, klass)


def _pipeline_from_spec(spec):
    steps = [(s, _load_class(c)(**spec.get(s, {}))) for s, c in spec['steps']]
    return Pipeline(steps)


def fit_classifier(reference_reads: types.GeneratorType,
                   reference_taxonomy: pd.Series,
                   classifier_specification: str, word_length: int=8
                   ) -> dict:
    spec = json.loads(classifier_specification)
    pipeline = _pipeline_from_spec(spec)
    params = {'word_length': word_length}
    pipeline = fit_pipeline(reference_reads, reference_taxonomy,
                            pipeline, **params)
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


def _register_fitter(name, spec):
    type_map = {int: Int, float: Float}  # add bool when available
    annotations = {}
    parameters = {}
    class_name = spec['steps'][-1][1]
    signature = inspect.signature(_load_class(class_name))
    for param_name, param in signature.parameters.items():
        if callable(param.default):  # callable introduces too many issues
            continue
        d_type = type(param.default)
        parameters[param_name] = type_map.get(d_type, Str)
        annotations[param_name] = d_type if d_type in type_map else str

    def _generic_fitter(reference_reads: types.GeneratorType,
                        reference_taxonomy: pd.Series,
                        word_length: int=8, **kwargs) -> dict:
        this_spec = copy.deepcopy(spec)
        for param in kwargs:
            try:
                kwargs[param] = json.loads(kwargs[param])
            except (json.JSONDecodeError, TypeError):
                pass
        this_spec[spec['steps'][-1][0]] = kwargs
        pipeline = _pipeline_from_spec(this_spec)
        params = {'word_length': word_length}
        pipeline = fit_pipeline(reference_reads, reference_taxonomy,
                                pipeline, **params)
        return {'params': params, 'pipeline': pipeline}

    parameters.update({'word_length': Int})
    _generic_fitter.__annotations__.update(annotations)
    _generic_fitter.__name__ = 'fit_classifier_' + name
    plugin.methods.register_function(
        function=_generic_fitter,
        inputs={'reference_reads': FeatureData[Sequence],
                'reference_taxonomy': ReferenceFeatures[SSU]},
        parameters=parameters,
        outputs=[('classifier', TaxonomicClassifier)],
        name='Train the ' + class_name + ' classifier.',
        description='Create a ' + class_name + ' classifier for reads'
    )
    _generic_fitter.__name__ = 'fit_classifier_' + name + '_paired_end'
    plugin.methods.register_function(
        function=_generic_fitter,
        inputs={'reference_reads': FeatureData[PairedEndSequence],
                'reference_taxonomy': ReferenceFeatures[SSU]},
        parameters=parameters,
        outputs=[('classifier', TaxonomicClassifier)],
        name='Train the ' + class_name + ' classifier.',
        description='Create a ' + class_name +
                    ' classifier for paired-end reads'
    )

for name, pipeline in _specific_fitters:
    _register_fitter(name, pipeline)
