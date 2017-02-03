# ----------------------------------------------------------------------------
# Copyright (c) 2016-2017, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import json
import importlib
import inspect
import warnings

import pandas as pd
from qiime2.plugin import Int, Str, Float, Bool
from q2_types.feature_data import FeatureData, Taxonomy, Sequence, DNAIterator
from sklearn.pipeline import Pipeline
import sklearn

from ._skl import fit_pipeline, predict, _specific_fitters
from ._taxonomic_classifier import TaxonomicClassifier
from .plugin_setup import plugin


def _load_class(classname):
    err_message = classname + ' is not a recognised class'
    if '.' not in classname:
        raise ValueError(err_message)
    module, klass = classname.rsplit('.', 1)
    if module == 'custom':
        module = importlib.import_module('.custom', 'q2_feature_classifier')
    elif importlib.util.find_spec('.'+module, 'sklearn') is not None:
        module = importlib.import_module('.'+module, 'sklearn')
    else:
        raise ValueError(err_message)
    if not hasattr(module, klass):
        raise ValueError(err_message)
    klass = getattr(module, klass)
    if not issubclass(klass, sklearn.base.BaseEstimator):
        raise ValueError(err_message)
    return klass


def spec_from_pipeline(pipeline):
    class StepsEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'get_params'):
                encoded = {}
                params = obj.get_params()
                subobjs = []
                for key, value in params.items():
                    if hasattr(value, 'get_params'):
                        subobjs.append(key + '__')

                for key, value in params.items():
                    for so in subobjs:
                        if key.startswith(so):
                            break
                    else:
                        if hasattr(value, 'get_params'):
                            encoded[key] = self.default(value)
                        try:
                            json.dumps(value, cls=StepsEncoder)
                            encoded[key] = value
                        except TypeError:
                            pass

                module = obj.__module__
                type = module + '.' + obj.__class__.__name__
                encoded['__type__'] = type.split('.', 1)[1]
                return encoded
            return json.JSONEncoder.default(self, obj)
    steps = pipeline.get_params()['steps']
    return json.loads(json.dumps(steps, cls=StepsEncoder))


def pipeline_from_spec(spec):
    def as_steps(obj):
        if '__type__' in obj:
            klass = _load_class(obj['__type__'])
            return klass(**{k: v for k, v in obj.items() if k != '__type__'})
        return obj

    steps = json.loads(json.dumps(spec), object_hook=as_steps)
    return Pipeline(steps)


def warn_about_sklearn():
    warning = (
        'The TaxonomicClassifier artifact that results from this method was '
        'trained using scikit-learn version %s. It cannot be used with other '
        'versions of scikit-learn. (While the classifier may complete '
        'successfully, the results will be unreliable.)' % sklearn.__version__)
    warnings.warn(warning, UserWarning)


def fit_classifier(reference_reads: DNAIterator, reference_taxonomy: pd.Series,
                   classifier_specification: str) -> Pipeline:
    warn_about_sklearn()
    spec = json.loads(classifier_specification)
    pipeline = pipeline_from_spec(spec)
    pipeline = fit_pipeline(reference_reads, reference_taxonomy, pipeline)
    return pipeline


plugin.methods.register_function(
    function=fit_classifier,
    inputs={'reference_reads': FeatureData[Sequence],
            'reference_taxonomy': FeatureData[Taxonomy]},
    parameters={'classifier_specification': Str},
    outputs=[('classifier', TaxonomicClassifier)],
    name='Train a scikit-learn classifier.',
    description='Train a scikit-learn classifier to classify reads.'
)


def classify(reads: DNAIterator, classifier: Pipeline,
             chunk_size: int=262144, n_jobs: int=1,
             pre_dispatch: str='2*n_jobs', confidence: float=-1.
             ) -> pd.DataFrame:
    predictions = predict(reads, classifier, chunk_size=chunk_size,
                          n_jobs=n_jobs, pre_dispatch=pre_dispatch,
                          confidence=confidence)
    result = list(zip(*predictions))
    if len(result) == 0:
        seq_ids, taxonomy, confidence = [], [], []
    else:
        seq_ids, taxonomy, confidence = result
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
    parameter_descriptions={'confidence': 'Confidence threshold for limiting '
                            'taxonomic depth. Negative value disables. '
                            'Currently experimental. USE WITH CAUTION.'}
)


def _pipeline_signature(spec):
    type_map = {int: Int, float: Float, bool: Bool, str: Str}
    parameters = {}
    signature_params = []
    pipeline = pipeline_from_spec(spec)
    params = pipeline.get_params()
    for param, default in sorted(params.items()):
        try:
            json.dumps(default)
        except TypeError:
            continue
        kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        if type(default) in type_map:
            annotation = type(default)
        else:
            annotation = str
            default = json.dumps(default)
        new_param = inspect.Parameter(param, kind, default=default,
                                      annotation=annotation)
        signature_params.append(new_param)
        parameters[param] = type_map.get(annotation, Str)
    return parameters, signature_params


def _register_fitter(name, spec):
    parameters, signature_params = _pipeline_signature(spec)

    def generic_fitter(reference_reads: DNAIterator,
                       reference_taxonomy: pd.Series, **kwargs) -> Pipeline:
        warn_about_sklearn()
        for param in kwargs:
            try:
                kwargs[param] = json.loads(kwargs[param])
            except (json.JSONDecodeError, TypeError):
                pass
        pipeline = pipeline_from_spec(spec)
        pipeline.set_params(**kwargs)
        pipeline = fit_pipeline(reference_reads, reference_taxonomy,
                                pipeline)
        return pipeline

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
        name='Train the ' + name + ' classifier.',
        description='Create a ' + name + ' classifier for reads'
    )


for name, pipeline in _specific_fitters:
    _register_fitter(name, pipeline)
