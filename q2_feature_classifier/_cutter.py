# ----------------------------------------------------------------------------
# Copyright (c) 2016-2017, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import Int, Str, Choices
from q2_types.feature_data import (
    FeatureData, PairedEndSequence, Sequence, DNAIterator, PairedDNAIterator)
import skbio

from ._gregex import extract_reads_by_match, extract_reads_by_position
from .plugin_setup import plugin


def extract_paired_end_reads(sequences: DNAIterator, read_length: int,
                             f_primer: str, r_primer: str,
                             method: str='position', n_sample: int=10000
                             ) -> PairedDNAIterator:
    """Extract the reads selected by a primer or primer pair.

    Parameters
    ----------
    sequences : list
        an aligned list of skbio.sequence.DNA query sequences
    readlength : int
        length of each read
    f_primer : skbio.sequence.DNA
        forward primer sequence
    r_primer : skbio.sequence.DNA
        reverse primer sequence
    method : str, optional
        how to extract the reads. Should be one of 'position' or 'match'. The
        former extracts the same region from all reads based on the primer
        positions in a subset. The former only extracts reads if a regex match
        is achieved on each query.
    endedness : str, optional
        paired-end ('pe'), single-end forward ('se'), or single-end reverse
        ('ser')
    sample : int, optional
        size of the subset to use to estimate positions for all. Only for
        'position' method

    Returns
    -------
    tuple
        containing the read or read pair (left, right) in skbio.sequence.DNA
        objects
    """

    f_primer = skbio.DNA(f_primer)
    r_primer = skbio.DNA(r_primer)

    if method == 'match':
        result = extract_reads_by_match(sequences, read_length, f_primer,
                                        r_primer, 'pe')
    elif method == 'position':
        result = extract_reads_by_position(sequences, read_length, f_primer,
                                           r_primer, 'pe', n_sample)
    else:
        raise ValueError(method + ' method not supported')

    return PairedDNAIterator(result)


plugin.methods.register_function(
    function=extract_paired_end_reads,
    inputs={'sequences': FeatureData[Sequence]},
    parameters={'read_length': Int,
                'f_primer': Str,
                'r_primer': Str,
                'method': Str % Choices(['match', 'position']),
                'n_sample': Int},
    outputs=[('reads', FeatureData[PairedEndSequence])],
    name='Extract reads from reference.',
    description='Extract sequencing-like reads from a reference database.'
)


def extract_reads(sequences: DNAIterator, read_length: int,
                  f_primer: str, r_primer: str, method: str='position',
                  direction: str='forward', n_sample: int=10000
                  ) -> DNAIterator:
    """Extract the read selected by a primer or primer pair.

    Parameters
    ----------
    sequences : list
        an aligned list of skbio.sequence.DNA query sequences
    readlength : int
        length of each read
    f_primer : skbio.sequence.DNA
        forward primer sequence
    r_primer : skbio.sequence.DNA
        reverse primer sequence
    method : str, optional
        how to extract the reads. Should be one of 'position' or 'match'. The
        former extracts the same region from all reads based on the primer
        positions in a subset. The former only extracts reads if a regex match
        is achieved on each query.
    direction: str, option
        'forward' or 'reverse'. Returns the right end of the amplicon if
        'reverse'
    sample : int, optional
        size of the subset to use to estimate positions for all. Only for
        'position' method

    Returns
    -------
    tuple
        containing the read or read pair (left, right) in skbio.sequence.DNA
        objects
    """

    f_primer = skbio.DNA(f_primer)
    r_primer = skbio.DNA(r_primer)

    endedness = {'forward': 'se', 'reverse': 'ser'}[direction]

    if method == 'match':
        result = extract_reads_by_match(sequences, read_length, f_primer,
                                        r_primer, endedness)
    elif method == 'position':
        result = extract_reads_by_position(sequences, read_length, f_primer,
                                           r_primer, endedness, n_sample)
    else:
        raise ValueError(method + ' method not supported')

    def read_seqs():
        for single_sequence_tuple in result:
            yield single_sequence_tuple[0]
    return DNAIterator(read_seqs())


plugin.methods.register_function(
    function=extract_reads,
    inputs={'sequences': FeatureData[Sequence]},
    parameters={'read_length': Int,
                'f_primer': Str,
                'r_primer': Str,
                'method': Str % Choices(['match', 'position']),
                'direction': Str % Choices(['forward', 'reverse']),
                'n_sample': Int},
    outputs=[('reads', FeatureData[Sequence])],
    name='Extract reads from reference.',
    description='Extract sequencing-like reads from a reference database.'
)
