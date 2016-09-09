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
    module = importlib.import_module(module)
    return getattr(module, klass)


def _pipeline_from_spec(spec):
    steps = [(s, _load_class(c)(**spec.get(s, {}))) for s, c in spec['steps']]
    return Pipeline(steps)


def _fit_classifier(reference_reads, reference_taxonomy,
                    classifier_specification, word_length, taxonomy_separator,
                    taxonomy_depth, multioutput):
    spec = json.loads(classifier_specification)
    pipeline = _pipeline_from_spec(spec)
    params = {'word_length': word_length,
              'taxonomy_separator': taxonomy_separator,
              'taxonomy_depth': taxonomy_depth,
              'multioutput': multioutput}
    pipeline = fit_pipeline(reference_reads, reference_taxonomy,
                            pipeline, **params)
    return {'params': params, 'pipeline': pipeline}


def fit_classifier(reference_reads: DNAIterator,
                   reference_taxonomy: pd.Series,
                   classifier_specification: str, word_length: int=8,
                   taxonomy_separator: str='', taxonomy_depth: int=-1,
                   multioutput: bool=False) -> dict:
    return _fit_classifier(reference_reads, reference_taxonomy,
                           classifier_specification, word_length,
                           taxonomy_separator, taxonomy_depth, multioutput)

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


def fit_classifier_paired_end(reference_reads: PairedDNAIterator,
                              reference_taxonomy: pd.Series,
                              classifier_specification: str,
                              word_length: int=8, taxonomy_separator: str='',
                              taxonomy_depth: int=-1,
                              multioutput: bool=False) -> dict:
    return _fit_classifier(reference_reads, reference_taxonomy,
                           classifier_specification, word_length,
                           taxonomy_separator, taxonomy_depth, multioutput)

plugin.methods.register_function(
    function=fit_classifier_paired_end,
    inputs={'reference_reads': FeatureData[PairedEndSequence],
            'reference_taxonomy': FeatureData[Taxonomy]},
    parameters={**{'classifier_specification': Str}, **_fitter_parameters},
    outputs=[('classifier', TaxonomicClassifier)],
    name='Train a scikit-learn classifier.',
    description='Train a scikit-learn classifier to classify paired end reads.'
)


def _classify(reads, classifier, chunk_size):
    predictions = predict(reads, classifier['pipeline'],
                          **classifier['params'])
    seq_ids, classifications = zip(*predictions)
    result = pd.Series(classifications, index=seq_ids)
    result.name = 'taxonomy'
    result.index.name = 'Feature ID'
    return result.to_frame()


def classify(reads: DNAIterator, classifier: dict,
             chunk_size: int=262144) -> pd.DataFrame:
    return _classify(reads, classifier, chunk_size)


plugin.methods.register_function(
    function=classify,
    inputs={'reads': FeatureData[Sequence],
            'classifier': TaxonomicClassifier},
    parameters={'chunk_size': Int},
    outputs=[('classification', FeatureData[Taxonomy])],
    name='Classify reads by taxon.',
    description='Classify reads by taxon using a fitted classifier.',
)


def classify_paired_end(reads: PairedDNAIterator,
                        classifier: dict,
                        chunk_size: int=262144) -> pd.DataFrame:
    return _classify(reads, classifier, chunk_size)


plugin.methods.register_function(
    function=classify_paired_end,
    inputs={'reads': FeatureData[PairedEndSequence],
            'classifier': TaxonomicClassifier},
    parameters={'chunk_size': Int},
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

    def _generic_fitter(reference_reads, reference_taxonomy, word_length,
                        taxonomy_separator, taxonomy_depth, multioutput,
                        **kwargs):
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

    def generic_fitter(reference_reads: DNAIterator,
                       reference_taxonomy: pd.Series,
                       word_length: int=8, taxonomy_separator: str='',
                       taxonomy_depth: int=-1, multioutput: bool=False,
                       **kwargs) -> dict:
        return _generic_fitter(reference_reads, reference_taxonomy,
                               word_length, taxonomy_separator, taxonomy_depth,
                               multioutput, **kwargs)

    def generic_fitter_paired_end(reference_reads: PairedDNAIterator,
                                  reference_taxonomy: pd.Series,
                                  word_length: int=8,
                                  taxonomy_separator: str='',
                                  taxonomy_depth: int=-1,
                                  multioutput: bool=False,
                                  **kwargs) -> dict:
        return _generic_fitter(reference_reads, reference_taxonomy,
                               word_length, taxonomy_separator, taxonomy_depth,
                               multioutput, **kwargs)

    parameters.update(_fitter_parameters)

    for fitter in (generic_fitter, generic_fitter_paired_end):
        generic_signature = inspect.signature(fitter)
        new_params = list(generic_signature.parameters.values())[:-1]
        new_params.extend(signature_params)
        return_annotation = generic_signature.return_annotation
        new_signature = inspect.Signature(parameters=new_params,
                                          return_annotation=return_annotation)
        setattr(fitter, '__signature__', new_signature)

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
    generic_fitter_paired_end.__name__ = 'fit_classifier_%s_paired_end' % name
    plugin.methods.register_function(
        function=generic_fitter_paired_end,
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
