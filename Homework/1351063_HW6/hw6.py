__author__ = 'Tran Tinh Chi'

import os
import collections
import operator

from MyIndexer import MyIndexer
from MyIndexer import WORD_PATTERN

def is_relevant(query_class_name, result, words_in_query, word_doc_frequency):
    r_class_name = result[0]
    r_file_name = result[1]
    if r_class_name == query_class_name:
        for w in words_in_query.keys():
            if word_doc_frequency[w][r_class_name + "_" + r_file_name] > 0:
                return True
    return False


def main():
    stop_words = "stopwords_en.txt"
    data_directory = "data"

    # The process of picking 5000 documents for testing is also included in the method 'index_data' of MyIndexer
    indexer = MyIndexer(stop_words)
    indexer.split_train_test_sets(data_directory)
    indexer.index_data()

    print " * Naive Bayes "
    nb_precision = indexer.test_classifier_naive_bayes()
    print "-----------------------------------------------"
    print " * K-Nearest "
    kn_precision = indexer.test_classifier_k_nearest()
    print "-----------------------------------------------"
    print "Result : "
    print "--- Naive Bayes"
    print "       Precision : " + str(nb_precision)
    print ""
    print "--- K-Nearest"
    print "       Precision : " + str(kn_precision)

if __name__ == "__main__":
    main()