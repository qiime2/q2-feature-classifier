# ----------------------------------------------------------------------------
# Copyright (c) 2016-2025, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import skbio
import os
from joblib import Parallel, delayed, effective_n_jobs

from qiime2.plugin import Int, Str, Float, Range, Choices
from q2_types.feature_data import (FeatureData, Sequence, DNAIterator,
                                   DNASequencesDirectoryFormat, DNAFASTAFormat)
from q2_feature_classifier._skl import _chunks
from q2_feature_classifier.classifier import _autotune_reads_per_batch

from .plugin_setup import plugin


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
        beg, end = match.start + len(f_primer), match.stop - len(r_primer)
        return seq[beg:end]
    except StopIteration:
        return None


def _align_primer(primer, seq, reverse=False):
    if reverse:
        primer = primer.reverse_complement()
    best_score = None
    for p in sorted([str(s) for s in primer.expand_degenerates()]):
        p = skbio.DNA(p)
        try:
            # perform pairwise semi-global alignment, such that gaps on the
            # ends of primer aren't scored but gaps on the ends of seq are
            # scored
            aln = skbio.alignment.pair_align_nucl(
                p, seq, mode='global',
                free_ends=[True, True, False, False], trim_ends=True)
            score = aln.score
            if best_score is None or score > best_score:
                best_score = score
                best_aln = aln
                best_primer = p
        except IndexError:
            # this is currently necessary as it seems that if all positions
            # are "ends" then `skbio.alignment.pair_align_nucl` fails with
            # an IndexError (e.g., rather than returning an alignment with a
            # shape of (2, 0)).
            # See https://github.com/scikit-bio/scikit-bio/issues/2279
            # This should be removed when that issue is addressed as we are
            # assuming an IndexError means a very poor alignment but we may be
            # masking other IndexErrors.
            best_score = 0.0
    if best_score == 0.0:
        return None, 0, len(primer)
    msa = skbio.TabularMSA.from_path_seqs(best_aln.paths[0],
                                          (best_primer, seq))

    if reverse:
        amplicon_pos = best_aln.paths[0].starts[1]
    else:
        amplicon_pos = best_aln.paths[0].stops[1]

    n_matches = msa[0].match_frequency(msa[1])
    aligned_length = msa.shape[1]

    return amplicon_pos, n_matches, aligned_length


def _approx_match(seq, f_primer, r_primer, identity):
    amp_start, f_matches, f_length = _align_primer(f_primer, seq)
    amp_end, r_matches, r_length = _align_primer(r_primer, seq, reverse=True)
    if f_matches == 0 or r_matches == 0:
        return None
    elif f_matches / f_length >= identity and r_matches / r_length >= identity:
        return seq[amp_start:amp_end]
    else:
        return None


def _gen_reads(sequence, f_primer, r_primer, trim_right, trunc_len, trim_left,
               identity, min_length, max_length, read_orientation):
    f_primer = skbio.DNA(f_primer)
    r_primer = skbio.DNA(r_primer)
    amp = None
    if read_orientation in ['forward', 'both']:
        amp = _exact_match(sequence, f_primer, r_primer)
    if not amp and read_orientation in ['reverse', 'both']:
        amp = _exact_match(sequence.reverse_complement(), f_primer, r_primer)
    if not amp and read_orientation in ['forward', 'both']:
        amp = _approx_match(sequence, f_primer, r_primer, identity)
    if not amp and read_orientation in ['reverse', 'both']:
        amp = _approx_match(
            sequence.reverse_complement(), f_primer, r_primer, identity)
    if not amp:
        return
    # we want to filter by max length before trimming
    if max_length > 0 and len(amp) > max_length:
        return
    if trim_right > 0:
        amp = amp[:-trim_right]
    if trunc_len > 0:
        amp = amp[:trunc_len]
    if trim_left > 0:
        amp = amp[trim_left:]
    if min_length > 0 and len(amp) < min_length:
        return
    if not amp:
        return
    return amp


def extract_reads(sequences: DNASequencesDirectoryFormat, f_primer: str,
                  r_primer: str, trim_right: int = 0,
                  trunc_len: int = 0, trim_left: int = 0,
                  identity: float = 0.8, min_length: int = 50,
                  max_length: int = 0, n_jobs: int = 1,
                  batch_size: int = 'auto', read_orientation: str = 'both') \
                  -> DNAFASTAFormat:
    """Extract the read selected by a primer or primer pair. Only sequences
    which match the primers at greater than the specified identity are returned

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
    with open(str(ff), 'a') as fh:
        with Parallel(n_jobs) as parallel:
            for chunk in _chunks(sequences, batch_size):
                amplicons = parallel(delayed(_gen_reads)(sequence, f_primer,
                                                         r_primer,
                                                         trim_right,
                                                         trunc_len,
                                                         trim_left,
                                                         identity,
                                                         min_length,
                                                         max_length,
                                                         read_orientation)
                                     for sequence in chunk)
                for amplicon in amplicons:
                    if amplicon is not None:
                        skbio.write(amplicon, format='fasta', into=fh)
    if os.stat(str(ff)).st_size == 0:
        raise RuntimeError("No matches found")
    return ff


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
    outputs=[('reads', FeatureData[Sequence])],
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
                            'directions.'}
)
