# ----------------------------------------------------------------------------
# Copyright (c) 2016-2017, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from itertools import chain
from numpy import empty, percentile
from skbio import DNA


def _seq_to_regex(seq):
    """Build a regex out of a IUPAC consensus sequence"""
    result = []
    for base in str(seq):
        if base in DNA.degenerate_chars:
            result.append('[{0}]'.format(
                ''.join(DNA.degenerate_map[base])))
        else:
            result.append(base)

    return ''.join(result)


def _primers_to_regex(f_primer, r_primer):
    return '({0}.*{1})'.format(_seq_to_regex(f_primer),
                               _seq_to_regex(r_primer.reverse_complement()))


def extract_reads_by_match(aln, readlength, f_primer, r_primer, endedness):
    regex = _primers_to_regex(f_primer, r_primer)
    for query in aln:
        query = query.degap()
        for match in query.find_with_regex(regex):
            reads = []
            lstart = match.start + len(f_primer)
            lstop = lstart + readlength
            rstop = match.stop - len(r_primer)
            rstart = max(rstop - readlength, 0)
            if endedness in ('pe', 'se'):
                reads.append(query[lstart:min(lstop, rstop)])
            if endedness in ('pe', 'ser'):
                reads.append(query[max(rstart, lstart):rstop])
            yield reads


def extract_reads_by_position(aln, readlength, f_primer, r_primer,
                              endedness, sample=10000):
    regex = _primers_to_regex(f_primer, r_primer)
    lstarts = empty(sample, dtype=int)
    lstops = empty(sample, dtype=int)
    rstarts = empty(sample, dtype=int)
    rstops = empty(sample, dtype=int)
    query_cache = []
    i = 0
    for query in aln:
        query_cache.append(query)
        gaps = query.gaps()
        for match in query.find_with_regex(regex, ignore=gaps):
            n = 0
            for j in range(match.start, len(query)):
                if gaps[j]:
                    continue
                n += 1
                if n == len(f_primer):
                    lstart = j + 1
                elif n == len(f_primer) + readlength:
                    lstop = j + 1
                    break
            else:
                lstop = j + 1
            n = 0
            for j in range(match.stop-1, -1, -1):
                if gaps[j]:
                    continue
                n += 1
                if n == len(r_primer):
                    rstop = j
                elif n == len(r_primer) + readlength:
                    rstart = j
                    break
            else:
                rstart = j

            if endedness in ('pe', 'se'):
                lstarts[i] = lstart
                lstops[i] = min(lstop, rstop)
            if endedness in ('pe', 'ser'):
                rstarts[i] = max(rstart, lstart)
                rstops[i] = rstop

            i += 1
        if i == sample:
            break

    if endedness in ('pe', 'se'):
        lstart = percentile(lstarts[:i], 50, interpolation='nearest')
        lstop = percentile(lstops[:i], 50, interpolation='nearest')
    if endedness in ('pe', 'ser'):
        rstart = percentile(rstarts[:i], 50, interpolation='nearest')
        rstop = percentile(rstops[:i], 50, interpolation='nearest')

    for query in chain(query_cache, aln):
        reads = []
        if endedness in ('pe', 'se'):
            reads.append(query[lstart:lstop].degap())
        if endedness in ('pe', 'ser'):
            reads.append(query[rstart:rstop].degap())
        yield reads
