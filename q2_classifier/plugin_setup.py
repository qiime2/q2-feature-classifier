from qiime.plugin import Plugin, Int, Properties, Str, Choices

import q2_classifier
from q2_types import (
    FeatureTable, Frequency, RelativeFrequency, PresenceAbsence,
    ReferenceFeatures, SSU, FeatureData, Taxonomy, Sequence)

plugin = Plugin(
    name='classifier',
    version=q2_classifier.__version__,
    website='https://github.com/BenKaehler/q2-classifier',
    package='q2_classifier'
)

plugin.methods.register_function(
    function=q2_classifier.classify,
    inputs={'sequences' : FeatureData[Sequence],
        'reference_sequences' : ReferenceFeatures[SSU],
        'reference_taxonomy' : ReferenceFeatures[SSU]},
    parameters={'depth': Int,
                'method' : Str % Choices(['naive-bayes', 'svc', 'perfect'])},
    outputs=[('classification', FeatureData[Taxonomy])],
    name='Train and apply feature classifier.',
    description='Train a classifier and apply it to feature data.'
)
