# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from collections import defaultdict


def train_assigner_perfect(reads, taxonomy):
    def s(read):
        return tuple(map(str, read))

    assigner = defaultdict(list)
    for read in reads:
        readid = read[0].metadata['id']
        assigner[s(read)].append(taxonomy.get(readid, 'unknown'))

    return lambda pair: assigner[s(pair)]
