# ----------------------------------------------------------------------------
# Copyright (c) 2016-2018, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import pandas as pd
from q2_types.feature_data import (
    FeatureData, Taxonomy, Sequence, DNAFASTAFormat)
from .plugin_setup import plugin, citations
from qiime2.plugin import Int, Str, Float, Choices, Range
from ._consensus_assignment import (_consensus_assignments,
                                    _get_default_unassignable_label)


def classify_consensus_vsearch(query: DNAFASTAFormat,
                               reference_reads: DNAFASTAFormat,
                               reference_taxonomy: pd.Series,
                               maxaccepts: int = 10,
                               perc_identity: int = 0.8, strand: str = 'both',
                               min_consensus: float = 0.51,
                               unassignable_label: str =
                               _get_default_unassignable_label(),
                               threads: str = 1) -> pd.DataFrame:
    seqs_fp = str(query)
    ref_fp = str(reference_reads)
    cmd = ['vsearch', '--usearch_global', seqs_fp, '--id', str(perc_identity),
           '--strand', strand, '--maxaccepts', str(maxaccepts),
           '--maxrejects', '0', '--output_no_hits', '--db', ref_fp,
           '--threads', str(threads), '--blast6out']
    consensus = _consensus_assignments(
        cmd, reference_taxonomy, min_consensus=min_consensus,
        unassignable_label=unassignable_label)
    return consensus


plugin.methods.register_function(
    function=classify_consensus_vsearch,
    inputs={'query': FeatureData[Sequence],
            'reference_reads': FeatureData[Sequence],
            'reference_taxonomy': FeatureData[Taxonomy]},
    parameters={'maxaccepts': Int % Range(1, None),
                'perc_identity': Float % Range(0.0, 1.0, inclusive_end=True),
                'strand': Str % Choices(['both', 'plus']),
                'min_consensus': Float % Range(0.5, 1.0, inclusive_end=True,
                                               inclusive_start=False),
                'unassignable_label': Str,
                'threads': Int},
    outputs=[('classification', FeatureData[Taxonomy])],
    input_descriptions={'query': 'Sequences to classify taxonomically.',
                        'reference_reads': 'reference sequences.',
                        'reference_taxonomy': 'reference taxonomy labels.'},
    parameter_descriptions={
        'strand': ('Align against reference sequences in forward ("plus") '
                   'or both directions ("both").'),
        'maxaccepts': ('Maximum number of hits to keep for each query. Must '
                       'be in range [0, infinity].'),
        'perc_identity': ('Reject match if percent identity to query is '
                          'lower. Must be in range [0.0, 1.0].'),
        'min_consensus': ('Minimum fraction of assignments must match top '
                          'hit to be accepted as consensus assignment. Must '
                          'be in range (0.5, 1.0].')
    },
    output_descriptions={'classification': 'The resulting taxonomy '
                         'classifications.'},
    name='VSEARCH consensus taxonomy classifier',
    description=('Assign taxonomy to query sequences using VSEARCH. Performs '
                 'VSEARCH global alignment between query and reference_reads, '
                 'then assigns consensus taxonomy to each query sequence from '
                 'among maxaccepts top hits, min_consensus of which share '
                 'that taxonomic assignment. Unlike classify-consensus-blast, '
                 'this method searches the entire reference database before '
                 'choosing the top N hits, not the first N hits.'),
    citations=[citations['rognes2016vsearch']]
)
