# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from collections import defaultdict, Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

def _kmer_dist(pair, word_length=8):
    """Combine the kmer_frequencies of a pair of sequences
    
    Parameters
    ----------
    pair : tuple
        a tuple of skbio.sequence.DNA sequences
    word_lenth : int
        kmer lengths
        
    Returns
    -------
    dict
        mapping from kmer to count
    """
    kmer_counts = Counter()
    for read in pair:
        if len(read) < word_length:
            continue
        freqs = read.kmer_frequencies(word_length)
        for kmer in freqs:
            kmer_counts[kmer] += freqs[kmer]
    return kmer_counts

def train_assigner_sklearn(reads, taxonomy, method='SVM'):
    """Manufacture a function to classify read pairs.
    
    Currently uses sklearn.svm.SVC. Will extend to provide
    other classifiers as time allows.
    
    Parameters
    ----------
    reads : list
        list of pairs of skbio.sequence.DNA reads
    taxonomy : dict
        mapping from taxon ids to taxonomic classifications
        
    Returns
    -------
    callable
        function which takes a read pair and returns a classification    
    """
    kmer_counts = {}
    for pair in reads:
        kmer_counts[pair[0].metadata['id']] = _kmer_dist(pair)
    seq_ids, kmer_counts = zip(*kmer_counts.items()) 
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(kmer_counts)
    y = [taxonomy.get(seq_id, 'unknown') for seq_id in seq_ids]
    selector = SelectPercentile()
    X = selector.fit_transform(X, y)

    if method == 'SVM':
        classifier = SVC(C=10, kernel='linear', degree=3,
                     gamma=0.001).fit(X, y)
    elif method == 'NB':
        classifier = MultinomialNB().fit(X, y)
    else:
        raise ValueError(method + ' method not supported')

    def assign(pair):
        kmer_counts = _kmer_dist(pair)
        x = selector.transform(vectorizer.transform(kmer_counts))
        return (classifier.predict(x)[0],)
    return assign
