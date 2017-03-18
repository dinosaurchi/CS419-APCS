__author__ = 'macpro'
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
    dict_file = "dict.txt"
    query_dir = "query"

    indexer = MyIndexer(stop_words)
    #indexer.index_data(data_directory)
    #indexer.export_dict(dict_file)

    indexer.import_dict(dict_file)

    dir = os.listdir(query_dir)
    if dir.count(".DS_Store") > 0:
        dir.remove(".DS_Store")
    data_dirs = os.listdir(data_directory)
    if data_dirs.count(".DS_Store") > 0:
        data_dirs.remove(".DS_Store")

    retrieval_docs = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    word_doc_frequency = indexer.get_word_doc_frequency()
    average_precision = collections.defaultdict(lambda: 0)

    for query_file_name in dir:
        temp = query_file_name.split("_")
        query_name = temp[0]
        query_class_name = temp[1]

        words_in_query = collections.defaultdict(lambda: 0)
        with open(query_dir + "/" + query_file_name, 'r') as f:
            import re
            words = re.findall(WORD_PATTERN, f.read(), re.VERBOSE)
            for w in words:
                w = indexer.normalize(w)
                if indexer.get_stop_wors()[w] == 0:
                    words_in_query[w] += 1

        count = 0
        for subdir_name in data_dirs:
            print "Analyzing query '"+ query_name + "' : " + str(float(count) / float(len(data_dirs)) * 100) + " %"
            count += 1

            path = data_directory + '/' + subdir_name
            data_files = os.listdir(path)

            c = 0
            for data_file_name in data_files:
                print "Processing : " + str(float(c) / float(len(data_files)) * 100) + " %"
                c += 1
                score = indexer.similarity(query_file_name, data_file_name, query_dir, path)
                if score > 0:
                    retrieval_docs[subdir_name][data_file_name] = score
            break

        print "OK!\n"

        print "Retrieval documents : ",
        result = sorted(retrieval_docs.items(), key=operator.itemgetter(1), reverse=True)
        relevant = 0
        cur_doc = 1
        for r in result:
            if is_relevant(query_class_name, r, words_in_query, word_doc_frequency):
                relevant += 1
                average_precision[query_name] += float(relevant / cur_doc)
            cur_doc += 1
        average_precision[query_name] /= relevant


if __name__ == "__main__":
    main()