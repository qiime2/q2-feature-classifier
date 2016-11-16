# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import Int, Str, Float
from q2_types.feature_data import (
    FeatureData, Sequence, DNAIterator)
import skbio

from .plugin_setup import plugin


def _seq_to_regex(seq):
    """Build a regex out of a IUPAC consensus sequence"""
    result = []
    for base in str(seq):
        if base in skbio.DNA.degenerate_chars:
            result.append('[{0}]'.format(
                ''.join(skbio.DNA.degenerate_map[base])))
        else:
            result.append(base)

    return ''.join(result)


def _primers_to_regex(f_primer, r_primer):
    return '({0}.*{1})'.format(_seq_to_regex(f_primer),
                               _seq_to_regex(r_primer.reverse_complement()))


def _aln_primer(primer, sequence, forward=True):
    if not forward:
        sequence = sequence.reverse_complement()

    # locally align the primer
    sm = skbio.alignment.make_identity_substitution_matrix(
        2, -3, alphabet=skbio.DNA.alphabet - skbio.DNA.gap_chars)
    dm = skbio.DNA.degenerate_map
    for c in skbio.DNA.degenerate_chars:
        for t in dm[c]:
            sm[c][t] = 2
            sm[t][c] = 2
    (aln_prim, aln_seq), score, (prim_pos, seq_pos) = \
        skbio.alignment.local_pairwise_align_ssw(primer, sequence)
    amplicon_pos = seq_pos[1]+len(primer)-prim_pos[1]

    # naively extend the alignment to be semi-global
    bits = [primer[:prim_pos[0]], aln_prim, primer[prim_pos[1]+1:]]
    aln_prim = ''.join(map(str, bits))
    bits = ['-'*(prim_pos[0]-seq_pos[0]),
            sequence[max(seq_pos[0]-prim_pos[0], 0):seq_pos[0]],
            aln_seq,
            sequence[seq_pos[1]+1:amplicon_pos],
            '-'*(amplicon_pos-len(sequence))]
    aln_seq = ''.join(map(str, bits))

    # count the matches
    matches = sum(a != '-' and s != '-' and sm[a][s] == 2
                  for a, s in zip(aln_prim, aln_seq))

    if not forward:
        amplicon_pos = len(sequence) - amplicon_pos

    return amplicon_pos, matches, len(aln_prim)


def _gen_reads(sequences: DNAIterator,  f_primer: str, r_primer: str,
               length: int=-1, identity: float=0.8) -> DNAIterator:
    f_primer = skbio.DNA(f_primer)
    r_primer = skbio.DNA(r_primer)
    regex = _primers_to_regex(f_primer, r_primer)
    for seq in sequences:
        try:
            # try exact match, because it's fast and usually works
            match = next(seq.find_with_regex(regex))
            beg, end = match.start + len(f_primer), match.stop - len(r_primer)
        except StopIteration:
            # try a little bit harder with a local alignment
            beg, b_matches, b_length = _aln_primer(f_primer, seq)
            end, e_matches, e_length = _aln_primer(r_primer, seq,
                                                   forward=False)
            if (b_matches + e_matches) / (b_length + e_length) < identity:
                continue
        if end - beg <= 0:
            continue
        if length <= 0:
            yield seq[beg:end]
        else:
            yield seq[beg:min(beg + length, end)]


def extract_reads(sequences: DNAIterator,  f_primer: str, r_primer: str,
                  length: int=-1, identity: float=0.8) -> DNAIterator:
    """Extract the read selected by a primer or primer pair. Only sequences
    which match the primers at greater than the specified identity are returned

    Parameters
    ----------
    sequences : DNAIterator
        an aligned list of skbio.sequence.DNA query sequences
    f_primer : skbio.sequence.DNA
        forward primer sequence
    r_primer : skbio.sequence.DNA
        reverse primer sequence
    length : int, optional
        length of each read. Full amplicon is returned if length is negative
    identity : float, optional
        minimum combined primer match identity threshold. Default: 0.8

    Returns
    -------
    q2_types.DNAIterator
        containing the reads
    """
    return DNAIterator(_gen_reads(sequences, f_primer, r_primer, length,
                                  identity))

plugin.methods.register_function(
    function=extract_reads,
    inputs={'sequences': FeatureData[Sequence]},
    parameters={'length': Int,
                'f_primer': Str,
                'r_primer': Str,
                'identity': Float},
    outputs=[('reads', FeatureData[Sequence])],
    name='Extract reads from reference.',
    description='Extract sequencing-like reads from a reference database.'
)
