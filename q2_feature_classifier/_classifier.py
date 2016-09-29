# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import json
import importlib
import inspect
import copy

import pandas as pd
from qiime.plugin import Int, Str, Float, Bool
from q2_types import (FeatureData, Taxonomy, Sequence, PairedEndSequence,
                      DNAIterator, PairedDNAIterator)
from sklearn.pipeline import Pipeline

from ._skl import fit_pipeline, predict, _specific_fitters
from ._taxonomic_classifier import TaxonomicClassifier
from .plugin_setup import plugin


def _load_class(classname):
    module, klass = classname.rsplit('.', 1)
    module = importlib.import_module('.'+module, 'sklearn')
    return getattr(module, klass)


def _pipeline_from_spec(spec):
    steps = [(s, _load_class(c)(**spec.get(s, {}))) for s, c in spec['steps']]
    return Pipeline(steps)


def fit_classifier(reference_reads: DNAIterator,
                   reference_taxonomy: pd.Series,
                   classifier_specification: str, word_length: int=8,
                   taxonomy_separator: str=';', taxonomy_depth: int=-1,
                   multioutput: bool=False) -> dict:
    spec = json.loads(classifier_specification)
    pipeline = _pipeline_from_spec(spec)
    params = {'word_length': word_length,
              'taxonomy_separator': taxonomy_separator,
              'taxonomy_depth': taxonomy_depth,
              'multioutput': multioutput}
    pipeline = fit_pipeline(reference_reads, reference_taxonomy,
                            pipeline, **params)
    return {'params': params, 'pipeline': pipeline}

_fitter_parameters = {'word_length': Int, 'taxonomy_separator': Str,
                      'taxonomy_depth': Int, 'multioutput': Bool}

plugin.methods.register_function(
    function=fit_classifier,
    inputs={'reference_reads': FeatureData[Sequence],
            'reference_taxonomy': FeatureData[Taxonomy]},
    parameters={**{'classifier_specification': Str}, **_fitter_parameters},
    outputs=[('classifier', TaxonomicClassifier)],
    name='Train a scikit-learn classifier.',
    description='Train a scikit-learn classifier to classify reads.'
)

fit_classifier.__name__ = 'fit_classifier_paired_end'
_fitter_signature = inspect.signature(fit_classifier)
_fitter_params = list(_fitter_signature.parameters.values())
_fitter_params[0] = inspect.Parameter(_fitter_params[0].name,
                                      _fitter_params[0].kind,
                                      annotation=PairedDNAIterator,
                                      default=_fitter_params[0].default)
_return_annotation = _fitter_signature.return_annotation
_fitter_signature = inspect.Signature(parameters=_fitter_params,
                                      return_annotation=_return_annotation)
fit_classifier.__signature__ = _fitter_signature

plugin.methods.register_function(
    function=fit_classifier,
    inputs={'reference_reads': FeatureData[PairedEndSequence],
            'reference_taxonomy': FeatureData[Taxonomy]},
    parameters={**{'classifier_specification': Str}, **_fitter_parameters},
    outputs=[('classifier', TaxonomicClassifier)],
    name='Train a scikit-learn classifier.',
    description='Train a scikit-learn classifier to classify paired end reads.'
)


def classify(reads: DNAIterator, classifier: dict,
             chunk_size: int=262144, n_jobs: int=1,
             pre_dispatch: str='2*n_jobs', confidence: float=-1.
             ) -> pd.DataFrame:
    predictions = predict(reads, classifier['pipeline'], chunk_size=chunk_size,
                          n_jobs=n_jobs, pre_dispatch=pre_dispatch,
                          confidence=confidence, **classifier['params'])
    seq_ids, taxonomy, confidence = zip(*predictions)
    result = pd.DataFrame({'Taxon': taxonomy, 'Confidence': confidence},
                          index=seq_ids, columns=['Taxon', 'Confidence'])
    result.index.name = 'Feature ID'
    return result

_classify_parameters = {'chunk_size': Int, 'n_jobs': Int, 'pre_dispatch': Str,
                        'confidence': Float}

plugin.methods.register_function(
    function=classify,
    inputs={'reads': FeatureData[Sequence],
            'classifier': TaxonomicClassifier},
    parameters=_classify_parameters,
    outputs=[('classification', FeatureData[Taxonomy])],
    name='Classify reads by taxon.',
    description='Classify reads by taxon using a fitted classifier.',
)

classify.__name__ = 'classify_paired_end'
_classify_signature = inspect.signature(classify)
_classify_params = list(_classify_signature.parameters.values())
_classify_params[0] = inspect.Parameter(_classify_params[0].name,
                                        _classify_params[0].kind,
                                        annotation=PairedDNAIterator,
                                        default=_classify_params[0].default)
_return_annotation = _classify_signature.return_annotation
_classify_signature = inspect.Signature(parameters=_classify_params,
                                        return_annotation=_return_annotation)
classify.__signature__ = _classify_signature

plugin.methods.register_function(
    function=classify,
    inputs={'reads': FeatureData[PairedEndSequence],
            'classifier': TaxonomicClassifier},
    parameters=_classify_parameters,
    outputs=[('classification', FeatureData[Taxonomy])],
    name='Classify reads by taxon.',
    description='Classify reads by taxon using a fitted classifier.',
)


def _register_fitter(name, spec):
    type_map = {int: Int, float: Float, bool: Bool, str: Str}
    parameters = {}
    signature_params = []
    class_name = spec['steps'][-1][1]
    defaults = spec[spec['steps'][-1][0]]
    signature = inspect.signature(_load_class(class_name))
    for param_name, parameter in signature.parameters.items():
        if callable(parameter.default):  # callable introduces too many issues
            continue
        default = defaults.get(param_name, parameter.default)
        annotation = type(default) if type(default) in type_map else str
        default = json.dumps(default) if annotation is str else default
        new_param = inspect.Parameter(param_name, parameter.kind,
                                      default=default, annotation=annotation)
        parameters[param_name] = type_map.get(annotation, Str)
        signature_params.append(new_param)

    def generic_fitter(reference_reads: DNAIterator,
                       reference_taxonomy: pd.Series,
                       word_length: int=8, taxonomy_separator: str=';',
                       taxonomy_depth: int=-1, multioutput: bool=False,
                       **kwargs) -> dict:
        this_spec = copy.deepcopy(spec)
        for param in kwargs:
            try:
                kwargs[param] = json.loads(kwargs[param])
            except (json.JSONDecodeError, TypeError):
                pass
        this_spec[spec['steps'][-1][0]] = kwargs
        pipeline = _pipeline_from_spec(this_spec)
        params = {'word_length': word_length,
                  'taxonomy_separator': taxonomy_separator,
                  'taxonomy_depth': taxonomy_depth,
                  'multioutput': multioutput}
        pipeline = fit_pipeline(reference_reads, reference_taxonomy,
                                pipeline, **params)
        return {'params': params, 'pipeline': pipeline}

    parameters.update(_fitter_parameters)

    generic_signature = inspect.signature(generic_fitter)
    new_params = list(generic_signature.parameters.values())[:-1]
    new_params.extend(signature_params)
    return_annotation = generic_signature.return_annotation
    new_signature = inspect.Signature(parameters=new_params,
                                      return_annotation=return_annotation)
    generic_fitter.__signature__ = new_signature
    generic_fitter.__name__ = 'fit_classifier_' + name
    plugin.methods.register_function(
        function=generic_fitter,
        inputs={'reference_reads': FeatureData[Sequence],
                'reference_taxonomy': FeatureData[Taxonomy]},
        parameters=parameters,
        outputs=[('classifier', TaxonomicClassifier)],
        name='Train the ' + class_name + ' classifier.',
        description='Create a ' + class_name + ' classifier for reads'
    )

    generic_fitter.__name__ = 'fit_classifier_%s_paired_end' % name
    new_params[0] = inspect.Parameter(new_params[0].name,
                                      new_params[0].kind,
                                      annotation=PairedDNAIterator,
                                      default=new_params[0].default)
    new_signature = inspect.Signature(parameters=new_params,
                                      return_annotation=return_annotation)
    generic_fitter.__signature__ = new_signature
    plugin.methods.register_function(
        function=generic_fitter,
        inputs={'reference_reads': FeatureData[PairedEndSequence],
                'reference_taxonomy': FeatureData[Taxonomy]},
        parameters=parameters,
        outputs=[('classifier', TaxonomicClassifier)],
        name='Train the ' + class_name + ' classifier.',
        description='Create a ' + class_name +
                    ' classifier for paired-end reads'
    )

for name, pipeline in _specific_fitters:
    _register_fitter(name, pipeline)
