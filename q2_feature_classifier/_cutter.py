# ----------------------------------------------------------------------------
# Copyright (c) 2016-2026, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import skbio
import os
import numpy as np
import pandas as pd
import qiime2
from collections import namedtuple
from joblib import Parallel, delayed, effective_n_jobs

from qiime2.plugin import Int, Str, Float, Range, Choices
from q2_types.feature_data import (FeatureData, Sequence, DNAIterator,
                                   DNASequencesDirectoryFormat, DNAFASTAFormat)
from q2_types.metadata import ImmutableMetadata, ImmutableMetadataFormat
from q2_feature_classifier._skl import _chunks
from q2_feature_classifier.classifier import _autotune_reads_per_batch

from .plugin_setup import plugin


_AlignResult = namedtuple(
    '_AlignResult', ['amplicon_pos', 'match_percent', 'primer_start', 'primer_end'])

_STATS_COLUMNS = [
    'outcome', 'match-orientation', 'match-method',
    'f-primer-start', 'f-primer-end', 'r-primer-start', 'r-primer-end',
    'f-primer-match-pct', 'r-primer-match-pct',
    'amplicon-length-pre-trim', 'amplicon-length-post-trim',
    'input-sequence-length',
]


def _seq_to_regex(seq):
    """Build a regex out of a IUPAC consensus sequence"""
    result = []
    for base in str(seq):
        if base in skbio.DNA.degenerate_chars:
            result.append('[{0}]'.format(
                ''.join(sorted(skbio.DNA.degenerate_map[base]))))
        else:
            result.append(base)

    return ''.join(result)


def _primers_to_regex(f_primer, r_primer):
    return '({0}.*{1})'.format(_seq_to_regex(f_primer),
                               _seq_to_regex(r_primer.reverse_complement()))


def _exact_match(seq, f_primer, r_primer):
    try:
        regex = _primers_to_regex(f_primer, r_primer)
        match = next(seq.find_with_regex(regex))
        f_start = match.start
        f_end = match.start + len(f_primer)
        r_start = match.stop - len(r_primer)
        r_end = match.stop
        return seq[f_end:r_start], f_start, f_end, r_start, r_end
    except StopIteration:
        return None


def _create_asymmetric_primer_substitution_matrix(match=2, mismatch=-3):
    """ Create an asymmetric substitution matrix for matching degenerate
        primers to target sequences.

        This is asymmetic such that degenerate characters in primers will
        score as matches when the target sequences contains a relevant
        character. Degenerate characters in target sequences however always
        score as a mismatch.

        This is designed on the assumption that primers contain degenerate
        characters because they represent a pool of sequences that will
        actually be present in a PCR reaction, but degenerate characters in
        target sequences represent error or uncertainty.

        This is intended for use with `skbio.alignment.pair_align`, and the
        primer should be passed as the first sequence and the target as the
        second sequence. This is because primers are represented by the rows
        and the target is represented by columns in the resulting
        skbio.SubstitutionMatrix.
    """
    definite_chars = sorted(skbio.DNA.definite_chars)
    degenerate_chars = sorted(skbio.DNA.degenerate_chars)
    chars = definite_chars + degenerate_chars

    sm = np.zeros((len(chars), len(chars)))

    for row, c1 in enumerate(chars):
        for col, c2 in enumerate(chars):
            if c1 in definite_chars:
                if c2 in definite_chars:
                    if c1 == c2:
                        sm[(row, col)] = match
                    else:
                        sm[(row, col)] = mismatch
                else:  # degenerate char in target sequence always mismatches
                    sm[(row, col)] = mismatch
            else:  # primer character is degenerate
                if c2 in skbio.DNA.degenerate_map[c1]:
                    sm[(row, col)] = match
                else:
                    sm[(row, col)] = mismatch
    return skbio.SubstitutionMatrix(chars, sm)


def _match_percent(primer, target):
    """ Compute proportion of matching positions in alignments, accounting for
        primer degeneracies.

        Parameters
        ----------
        primer : skbio.DNA
        target : skbio.DNA
    """
    matches = 0
    for primer_c, target_c in zip(str(primer), str(target)):
        if target_c in skbio.DNA.degenerate_chars:
            continue
        if primer_c == target_c:
            matches += 1
        elif (primer_c in skbio.DNA.degenerate_chars and
              target_c in skbio.DNA.degenerate_map[primer_c]):
            matches += 1
    return matches / len(primer)


def _align_primer(primer, target, substitution_matrix, reverse=False):
    if reverse:
        primer = primer.reverse_complement()

    # perform pairwise semi-global alignment such that gaps on the
    # ends of primer are free from penalization but gaps on the ends of target
    # are penalized. for example:

    # gaps on the ends of the primer, as in the following, are free:
    # --AAAA----------
    # CCAAAAGGGGCCCCTT
    # or
    # ----------CCCC--
    # CCAAAAGGGGCCCCTT

    # gaps on the end of the target, as in the following, incur the penalty:
    # AAAA------
    # --AAGGGGCC
    # or
    # ------CCCC
    # AAGGGGCC--

    # degenerate characters in primer match the characters they represent,
    # but degenerate characters in target are always considered
    # mismatches

    aln = skbio.alignment.pair_align_nucl(
        primer, target, mode='global', sub_score=substitution_matrix,
        free_ends=[True, True, False, False], trim_ends=True)
    msa = skbio.TabularMSA.from_path_seqs(aln.paths[0], (primer, target))
    match_percent = _match_percent(msa[0], msa[1])

    primer_start = aln.paths[0].starts[1]
    primer_end = aln.paths[0].stops[1]

    if reverse:
        amplicon_pos = primer_start
    else:
        amplicon_pos = primer_end

    return _AlignResult(amplicon_pos, match_percent, primer_start, primer_end)


def _approx_match(seq, f_primer, r_primer, identity):
    substitution_matrix = _create_asymmetric_primer_substitution_matrix()
    f_result = _align_primer(f_primer, seq, substitution_matrix)
    r_result = _align_primer(r_primer, seq, substitution_matrix, reverse=True)
    if f_result.match_percent >= identity and r_result.match_percent >= identity:
        return (seq[f_result.amplicon_pos:r_result.amplicon_pos],
                f_result.primer_start, f_result.primer_end,
                r_result.primer_start, r_result.primer_end,
                f_result.match_percent, r_result.match_percent)
    else:
        return None


def _gen_reads(sequence, f_primer, r_primer, trim_right, trunc_len, trim_left,
               identity, min_length, max_length, read_orientation):
    f_primer = skbio.DNA(f_primer)
    r_primer = skbio.DNA(r_primer)

    stats = {
        'id': sequence.metadata['id'],
        'input-sequence-length': len(sequence),
        'match-orientation': 'none',
        'match-method': 'none',
        'f-primer-start': None,
        'f-primer-end': None,
        'r-primer-start': None,
        'r-primer-end': None,
        'f-primer-match-pct': None,
        'r-primer-match-pct': None,
        'amplicon-length-pre-trim': None,
        'amplicon-length-post-trim': None,
        'outcome': 'no-primer-match',
    }

    amp = None

    if read_orientation in ['forward', 'both']:
        result = _exact_match(sequence, f_primer, r_primer)
        if result is not None:
            amp, f_start, f_end, r_start, r_end = result
            stats.update({
                'match-orientation': 'forward', 'match-method': 'exact',
                'f-primer-start': f_start, 'f-primer-end': f_end,
                'r-primer-start': r_start, 'r-primer-end': r_end,
                'f-primer-match-pct': 1.0, 'r-primer-match-pct': 1.0,
            })

    if amp is None and read_orientation in ['reverse', 'both']:
        result = _exact_match(sequence.reverse_complement(), f_primer, r_primer)
        if result is not None:
            amp, f_start, f_end, r_start, r_end = result
            stats.update({
                'match-orientation': 'reverse', 'match-method': 'exact',
                'f-primer-start': f_start, 'f-primer-end': f_end,
                'r-primer-start': r_start, 'r-primer-end': r_end,
                'f-primer-match-pct': 1.0, 'r-primer-match-pct': 1.0,
            })

    if amp is None and read_orientation in ['forward', 'both']:
        result = _approx_match(sequence, f_primer, r_primer, identity)
        if result is not None:
            amp, f_start, f_end, r_start, r_end, f_pct, r_pct = result
            stats.update({
                'match-orientation': 'forward', 'match-method': 'approximate',
                'f-primer-start': f_start, 'f-primer-end': f_end,
                'r-primer-start': r_start, 'r-primer-end': r_end,
                'f-primer-match-pct': f_pct, 'r-primer-match-pct': r_pct,
            })

    if amp is None and read_orientation in ['reverse', 'both']:
        result = _approx_match(
            sequence.reverse_complement(), f_primer, r_primer, identity)
        if result is not None:
            amp, f_start, f_end, r_start, r_end, f_pct, r_pct = result
            stats.update({
                'match-orientation': 'reverse', 'match-method': 'approximate',
                'f-primer-start': f_start, 'f-primer-end': f_end,
                'r-primer-start': r_start, 'r-primer-end': r_end,
                'f-primer-match-pct': f_pct, 'r-primer-match-pct': r_pct,
            })

    if amp is None:
        return None, stats

    stats['amplicon-length-pre-trim'] = len(amp)

    # filter by max length before trimming
    if max_length > 0 and len(amp) > max_length:
        stats['outcome'] = 'excluded-max-length'
        return None, stats

    if trim_right > 0:
        amp = amp[:-trim_right]
    if trunc_len > 0:
        amp = amp[:trunc_len]
    if trim_left > 0:
        amp = amp[trim_left:]

    if len(amp) == 0:
        stats['outcome'] = 'excluded-empty-after-trim'
        return None, stats

    if min_length > 0 and len(amp) < min_length:
        stats['outcome'] = 'excluded-min-length'
        return None, stats

    stats['amplicon-length-post-trim'] = len(amp)
    stats['outcome'] = 'extracted'
    return amp, stats


def extract_reads(sequences: DNASequencesDirectoryFormat, f_primer: str,
                  r_primer: str, trim_right: int = 0,
                  trunc_len: int = 0, trim_left: int = 0,
                  identity: float = 0.7, min_length: int = 50,
                  max_length: int = 0, n_jobs: int = 1,
                  batch_size: int = 'auto', read_orientation: str = 'both') \
                  -> tuple[DNAFASTAFormat, ImmutableMetadataFormat]:
    """Extract the read selected by a primer or primer pair. Only sequences
    which match the primers at greater than the specified identity are
    returned. Note that the primers are *not* included in the extracted reads.

    Parameters
    ----------
    sequences : DNASequencesDirectoryFormat
        An aligned list of skbio.sequence.DNA query sequences
    f_primer : skbio.sequence.DNA
        Forward primer sequence
    r_primer : skbio.sequence.DNA
        Reverse primer sequence
    trim_right : int, optional
        `trim_right` nucleotides are removed from the 3' end if trim_right is
        positive. Applied before trunc_len.
    trunc_len : int, optional
        Read is cut to trunc_len if trunc_len is positive. Applied after
        trim_right.
    trim_left : int, optional
        `trim_left` nucleotides are removed from the 5' end if trim_left is
        positive. Applied after trim_right and trunc_len.
    identity : float, optional
        Minimum combined primer match identity threshold. Default: 0.8
    min_length: int, optional
        Minimum amplicon length. Shorter amplicons are discarded. Default: 50
    max_length: int, optional
        Maximum amplicon length. Longer amplicons are discarded.
    n_jobs: int, optional
        Number of seperate processes to break the task into.
    batch_size: int, optional
        Number of samples to be processed in one batch.
    read_orientation: str, optional
        'Orientation of primers relative to the sequences: "forward" searches '
        'for primer hits in the forward direction, "reverse" searches the '
        'reverse-complement, and "both" searches both directions.'
    Returns
    -------
    q2_types.DNAFASTAFormat
        containing the reads
    """
    if min_length > trunc_len - (trim_left + trim_right) and trunc_len > 0:
        raise ValueError('The minimum length setting is greater than the '
                         'length of the truncated sequences. This will cause '
                         'all sequences to be removed from the dataset. To '
                         'proceed, set '
                         'min_length ≤ trunc_len - (trim_left  + '
                         'trim_right).')

    n_jobs = effective_n_jobs(n_jobs)
    if batch_size == 'auto':
        batch_size = _autotune_reads_per_batch(
            sequences.file.view(DNAFASTAFormat), n_jobs)
    sequences = sequences.file.view(DNAIterator)
    ff = DNAFASTAFormat()
    all_stats = []
    with open(str(ff), 'a') as fh:
        with Parallel(n_jobs) as parallel:
            for chunk in _chunks(sequences, batch_size):
                results = parallel(delayed(_gen_reads)(sequence, f_primer,
                                                       r_primer,
                                                       trim_right,
                                                       trunc_len,
                                                       trim_left,
                                                       identity,
                                                       min_length,
                                                       max_length,
                                                       read_orientation)
                                   for sequence in chunk)
                for amplicon, stats in results:
                    all_stats.append(stats)
                    if amplicon is not None:
                        skbio.write(amplicon, format='fasta', into=fh)
    if os.stat(str(ff)).st_size == 0:
        raise RuntimeError("No matches found")
    stats_df = pd.DataFrame(all_stats, columns=['id'] + _STATS_COLUMNS)
    stats_df = stats_df.set_index('id')
    stats_ff = ImmutableMetadataFormat()
    qiime2.Metadata(stats_df).save(str(stats_ff))
    return ff, stats_ff


plugin.methods.register_function(
    function=extract_reads,
    inputs={'sequences': FeatureData[Sequence]},
    parameters={'trunc_len': Int,
                'trim_left': Int,
                'trim_right': Int,
                'f_primer': Str,
                'r_primer': Str,
                'identity': Float,
                'min_length': Int % Range(0, None),
                'max_length': Int % Range(0, None),
                'n_jobs': Int % Range(1, None),
                'batch_size': Int % Range(1, None) | Str % Choices(['auto']),
                'read_orientation': Str % Choices(['both', 'forward',
                                                   'reverse'])},
    outputs=[('reads', FeatureData[Sequence]),
             ('read_extraction_stats', ImmutableMetadata)],
    name='Extract reads from reference sequences.',
    description='Extract simulated amplicon reads from a reference database. '
                'Performs in-silico PCR to extract simulated amplicons from '
                'reference sequences that match the input primer sequences '
                '(within the mismatch threshold specified by `identity`). '
                'Both primer sequences must be in the 5\' -> 3\' orientation. '
                'Sequences that fail to match both primers will be excluded. '
                'Reads are extracted, trimmed, and filtered in the following '
                'order: 1. reads are extracted in specified orientation; 2. '
                'primers are removed; 3. reads longer than `max_length` are '
                'removed; 4. reads are trimmed with `trim_right`; 5. reads '
                'are truncated to `trunc_len`; 6. reads are trimmed with '
                '`trim_left`; 7. reads shorter than `min_length` are removed.',
    parameter_descriptions={
        'f_primer': 'forward primer sequence (5\' -> 3\').',
        'r_primer': 'reverse primer sequence (5\' -> 3\'). Do not use reverse-'
                    'complemented primer sequence.',
        'trim_right': 'trim_right nucleotides are removed from the 3\' end if '
                      'trim_right is positive. Applied before trunc_len and '
                      'trim_left.',
        'trunc_len': 'read is cut to trunc_len if trunc_len is positive. '
                     'Applied after trim_right but before trim_left.',
        'trim_left': 'trim_left nucleotides are removed from the 5\' end if '
                     'trim_left is positive. Applied after trim_right and '
                     'trunc_len.',
        'identity': 'minimum combined primer match identity threshold.',
        'min_length': 'Minimum amplicon length. Shorter amplicons are '
                      'discarded. Applied after trimming and truncation, so '
                      'be aware that trimming may impact sequence retention. '
                      'Set to zero to disable min length filtering.',
        'max_length': 'Maximum amplicon length. Longer amplicons are '
                      'discarded. Applied before trimming and truncation, '
                      'so plan accordingly. Set to zero (default) to disable '
                      'max length filtering.',
        'n_jobs': 'Number of seperate processes to run.',
        'batch_size': 'Number of sequences to process in a batch. The `auto` '
                      'option is calculated from the number of sequences and '
                      'number of jobs specified.',
        'read_orientation': 'Orientation of primers relative to the '
                            'sequences: "forward" searches for primer hits in '
                            'the forward direction, "reverse" searches '
                            'reverse-complement, and "both" searches both '
                            'directions.'},
    output_descriptions={
        'reads': 'Extracted reads.',
        'read_extraction_stats': 'Per-input-sequence report of primer '
                                 'alignment outcomes. Includes match '
                                 'orientation, match method (exact or '
                                 'approximate), primer binding positions in '
                                 'the matched-orientation sequence, forward '
                                 'and reverse primer match percentages, '
                                 'amplicon length before and after trimming, '
                                 'and the final extraction outcome for each '
                                 'sequence.',
    }
)
