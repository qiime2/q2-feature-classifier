# ----------------------------------------------------------------------------
# Copyright (c) 2016-2017, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import pandas as pd
from q2_types.feature_data import (
    FeatureData, Taxonomy, Sequence, DNAFASTAFormat)
from qiime2.plugin import Int, Str, Float, Choices, Range
from .plugin_setup import plugin
from ._consensus_assignment import (_consensus_assignments,
                                    _get_default_unassignable_label)


def classify_consensus_blast(query: DNAFASTAFormat,
                             reference_reads: DNAFASTAFormat,
                             reference_taxonomy: pd.Series, maxaccepts: int=10,
                             perc_identity: float=0.8, strand: str='both',
                             evalue: float=0.001, min_consensus: float=0.51,
                             unassignable_label: str=
                             _get_default_unassignable_label(),
                             num_threads: str=1) -> pd.DataFrame:
    perc_identity = perc_identity * 100
    seqs_fp = str(query)
    ref_fp = str(reference_reads)
    cmd = ['blastn', '-query', seqs_fp, '-evalue', str(evalue), '-strand',
           strand, '-outfmt', '7', '-subject', ref_fp, '-perc_identity',
           str(perc_identity), '-max_target_seqs', str(maxaccepts),
           '-num_threads', str(num_threads), '-out']
    consensus = _consensus_assignments(
        cmd, reference_taxonomy, unassignable_label=unassignable_label,
        min_consensus=min_consensus, output_no_hits=True,
        exp_seq_ids=seqs_fp)

    return consensus


plugin.methods.register_function(
    function=classify_consensus_blast,
    inputs={'query': FeatureData[Sequence],
            'reference_reads': FeatureData[Sequence],
            'reference_taxonomy': FeatureData[Taxonomy]},
    parameters={'evalue': Float,
                'maxaccepts': Int % Range(1, None),
                'perc_identity': Float % Range(0.0, 1.0, inclusive_end=True),
                'strand': Str % Choices(['both', 'plus', 'minus']),
                'min_consensus': Float % Range(0.5, 1.0, inclusive_end=True,
                                               inclusive_start=False),
                'unassignable_label': Str,
                'num_threads': Int},
    outputs=[('classification', FeatureData[Taxonomy])],
    input_descriptions={'query': 'Sequences to classify taxonomically.',
                        'reference_reads': 'reference sequences.',
                        'reference_taxonomy': 'reference taxonomy labels.'},
    parameter_descriptions={
        'evalue': 'BLAST expectation value (E) threshold for saving hits.',
        'strand': ('Align against reference sequences in forward ("plus"), '
                   'reverse ("minus"), or both directions ("both").'),
        'maxaccepts': ('Maximum number of hits to keep for each query. Must '
                       'be in range [0, infinity].'),
        'perc_identity': ('Reject match if percent identity to query is '
                          'lower. Must be in range [0.0, 1.0].'),
        'min_consensus': ('Minimum fraction of assignments must match top '
                          'hit to be accepted as consensus assignment. Must '
                          'be in range (0.5, 1.0].')
    },
    output_descriptions={
        'classification': 'Taxonomy classifications of query sequences.'},
    name='BLAST+ consensus taxonomy classifier',
    description=('Assign taxonomy to query sequences using BLAST+. Performs '
                 'BLAST+ local alignment between query and reference_reads, '
                 'then assigns consensus taxonomy to each query sequence from '
                 'among maxaccepts top hits, min_consensus of which share '
                 'that taxonomic assignment.')
)
