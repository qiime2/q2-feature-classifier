# ----------------------------------------------------------------------------
# Copyright (c) 2016-2017, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import tempfile
import pandas as pd
from os.path import join
from q2_types.feature_data import (
    FeatureData, Taxonomy, Sequence, DNAFASTAFormat)
from qiime2.plugin import Int, Str, Float
from .plugin_setup import plugin
from ._consensus_assignment import (_consensus_assignments,
                                    _parse_params,
                                    _run_command)


def blast(query: DNAFASTAFormat, reference_reads: DNAFASTAFormat,
          reference_taxonomy: pd.Series, maxaccepts: int=10, min_id: int=0.8,
          strand: str='both', evalue: float=0.001, t_delim: str=';',
          min_consensus: float=0.51, unassignable_label: str="Unassigned",
          num_threads: str=1, params: str=None) -> pd.DataFrame:

    min_id = min_id * 100
    params = _parse_params(params)
    seqs_fp = str(query)
    ref_fp = str(reference_reads)
    with tempfile.TemporaryDirectory() as blastdir:
        blastref = join(blastdir, 'blastref')
        # make temporary blast db
        cmd = ['makeblastdb', '-in', ref_fp, '-out', blastref, '-dbtype',
               'nucl']
        _run_command(cmd)
        # execute blastn search
        cmd = ['blastn', '-query', seqs_fp, '-evalue', str(evalue), '-strand',
               strand, '-outfmt', '7', '-db', blastref, '-perc_identity',
               str(min_id), '-max_target_seqs', str(maxaccepts),
               '-num_threads', str(num_threads), *params, '-out']
        # consensus taxonomy assignment
        consensus = _consensus_assignments(
            cmd, reference_taxonomy, unassignable_label=unassignable_label,
            t_delim=t_delim, min_consensus=min_consensus)

    return consensus


plugin.methods.register_function(
    function=blast,
    inputs={'query': FeatureData[Sequence],
            'reference_reads': FeatureData[Sequence],
            'reference_taxonomy': FeatureData[Taxonomy]},
    parameters={'evalue': Float, 'maxaccepts': Int, 'min_id': Float,
                'strand': Str, 't_delim': Str, 'params': Str,
                'min_consensus': Float, 'unassignable_label': Str,
                'num_threads': Int},
    outputs=[('classification', FeatureData[Taxonomy])],
    name='Assign taxonomy using BLAST+.',
    description='Assign taxonomy to query sequences using BLAST+.',
    parameter_descriptions={'evalue': 'BLAST expectation value (E) threshold'
                            'for saving hits.', 'strand': 'plus|both',
                            'maxaccepts': 'Maximum number of hits to keep for'
                            'each query', 'min_id': 'Reject match if percent'
                            'identity to query is lower. Range [0.0 - 1.0]',
                            'min_consensus': 'Minimum fraction of assignments'
                            'must match top hit to be accepted as consensus'
                            'assignment. Must be in range [0.5 - 1.0]',
                            'params': 'comma-separated list of additional '
                            'command-line parameters and their values to use '
                            'during assignment. Parameters and their values '
                            'must be passed consecutively, and entered exactly'
                            ' as they should be passed to command line,'
                            'without spaces. E.g., "-evalue,10,-strand,both".'
                            }
)
