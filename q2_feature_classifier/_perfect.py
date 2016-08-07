from collections import defaultdict


def train_assigner_perfect(reads, taxonomy):
    def s(read):
        return tuple(map(str, read))
    
    assigner = defaultdict(list)
    for read in reads:
        readid = read[0].metadata['id']
        assigner[s(read)].append(taxonomy.get(readid, 'unknown'))

    return lambda pair: assigner[s(pair)]
