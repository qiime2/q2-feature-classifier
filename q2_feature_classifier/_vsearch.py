# ----------------------------------------------------------------------------
# Copyright (c) 2016-2017, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import pandas as pd
from ._consensus_assignment import (_consensus_assignments,
                                    _get_default_unassignable_label,
                                    _validate_params)
from q2_types.feature_data import (
    FeatureData, Taxonomy, Sequence, DNAFASTAFormat)
from .plugin_setup import plugin
from qiime2.plugin import Int, Str, Float


def vsearch(query: DNAFASTAFormat, reference_reads: DNAFASTAFormat,
            reference_taxonomy: pd.Series, maxaccepts: int=10, min_id: int=0.8,
            strand: str='both', min_consensus: float=0.51,
            unassignable_label: str=_get_default_unassignable_label(),
            num_threads: str=1) -> pd.DataFrame:

    _validate_params(min_id, maxaccepts, min_consensus)
    seqs_fp = str(query)
    ref_fp = str(reference_reads)
    cmd = ['vsearch', '--usearch_global', seqs_fp, '--id', str(min_id),
           '--strand', strand, '--maxaccepts', str(maxaccepts),
           '--maxrejects', '0', '--output_no_hits', '--db', ref_fp,
           '--threads', str(num_threads), '--blast6out']
    consensus = _consensus_assignments(
        cmd, reference_taxonomy, min_consensus=min_consensus,
        unassignable_label=unassignable_label)
    return consensus


plugin.methods.register_function(
    function=vsearch,
    inputs={'query': FeatureData[Sequence],
            'reference_reads': FeatureData[Sequence],
            'reference_taxonomy': FeatureData[Taxonomy]},
    parameters={'maxaccepts': Int, 'min_id': Float, 'strand': Str,
                'min_consensus': Float, 'unassignable_label': Str,
                'num_threads': Int},
    outputs=[('classification', FeatureData[Taxonomy])],
    input_descriptions={'query': 'Sequences to classify taxonomically.',
                        'reference_reads': 'reference sequences.',
                        'reference_taxonomy': 'reference taxonomy labels.'},
    parameter_descriptions={
        'strand': 'plus|both',
        'maxaccepts': ('Maximum number of hits to keep for each query. Must be'
                       '>= 1'),
        'min_id': 'Reject match if percent identity to query is lower. Range '
        '[0.0 - 1.0]',
        'min_consensus': ('Minimum fraction of assignments must match top hit'
                          'to be accepted as consensus assignment. Must be in'
                          'range [0.51 - 1.0]')
    },
    output_descriptions={'classification': 'The resulting taxonomy '
                         'classifications.'},
    name='VSEARCH consensus taxonomy classifier',
    description='Assign taxonomy to query sequences using VSEARCH.'
)
