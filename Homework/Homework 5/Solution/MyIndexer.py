__author__ = 'macpro'
import re
import os
import collections
from PorterStemmer import PorterStemmer
import math

WORD_PATTERN = r'\b[a-zA-Z]+-?[a-zA-Z]+|\b[1-9]{1,3}\.?[0-9]{1,3}'

class MyIndexer:
    def __init__(self, stop_words_file=""):
        self.word_doc_frequency = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        self.doc_class_frequency = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        self.total_words = 0

        self.stop_words = collections.defaultdict(lambda: False)
        self.stemmer = PorterStemmer()

        if stop_words_file != "":
            with open(stop_words_file, 'r') as f:
                for line in f:
                    for w in line.split():
                        w = self.normalize(w)
                        self.stop_words[w] = True

    def get_doc_class_frequency(self):
        return self.doc_class_frequency

    def get_stop_wors(self):
        return self.stop_words

    def get_word_doc_frequency(self):
        return self.word_doc_frequency

    def index_data(self, data_directory):
        dirs = os.listdir(data_directory)
        if dirs.count(".DS_Store") > 0:
            dirs.remove(".DS_Store")

        for subdir_name in dirs:
            path = data_directory + '/' + subdir_name
            files_name = os.listdir(path)
            if files_name.count(".DS_Store") > 0:
                files_name.remove(".DS_Store")

            print "Reading directory : " + subdir_name + " ... ",
            for f_name in files_name:
                self.doc_class_frequency[f_name][subdir_name] += 1
                with open(path + '/' + f_name, 'r') as f:
                    flag = False
                    for line in f:
                        if line[0] == ">" or line[0] == "\n":
                            flag = True

                        if flag is False:
                            continue

                        words = re.findall(WORD_PATTERN, line, re.VERBOSE)

                        for w in words:
                            w = self.normalize(w)
                            if not self.stop_words[w]:
                                self.word_doc_frequency[w][subdir_name + "_" + f_name] += 1
                                self.total_words += 1
            print " OK!"

    def import_dict(self, file_name):
        with open(file_name, 'r') as f:
            self.word_doc_frequency = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
            self.total_words = 0

            print "Importing Dict ... ",
            for line in f:
                words = line.split()
                w = words[0]
                n = int(words[1])
                self.total_words += n

                for i in range(2, n + 2):
                    d = words[i]
                    self.word_doc_frequency[w][d] += 1

            print "OK!"

    def export_dict(self, file_name):
        with open(file_name, 'w') as f:
            for w in sorted(self.word_doc_frequency.iterkeys()):
                if len(w) > 0:
                    f.write(w + " " + str(len(self.word_doc_frequency[w])))
                    for d in self.word_doc_frequency[w]:
                        f.write(" " + d)
                    f.write(" \n")

    def normalize(self, w):
        w = w.lower()
        w = self.stemmer.stem(w, 0, len(w) - 1)

        return w

    def get_table_frequencies_from_doc(self, doc_name, directory_path):
        count_frequencies = collections.defaultdict(lambda: 0)
        with open(directory_path + "/" + doc_name, 'r') as f:
            words = re.findall(WORD_PATTERN, f.read(), re.VERBOSE)
            for w in words:
                w = self.normalize(w)
                if self.stop_words[w] == 0:
                    count_frequencies[w] += 1

        return count_frequencies

    def get_frequency_from_doc(self, normalized_term, doc_name, directory_path):
        count_frequency = 0
        with open(directory_path + "/" + doc_name, 'r') as f:
            words = re.findall(WORD_PATTERN, f.read(), re.VERBOSE)
            for w in words:
                w = self.normalize(w)
                if w == normalized_term and self.stop_words[w] == 0:
                    count_frequency += 1

        return count_frequency

    def tf(self, normalized_term, doc_name, directory_path=""):
        if self.stop_words[normalized_term] > 0:
            return 0

        c = directory_path.split("/")[1]
        f = self.word_doc_frequency[normalized_term][c + "_" + doc_name]
        if f > 0:
            return 1 + math.log(f)

        # # This is the case when doc is query. Because we did not include the query in the train set, thus we do not
        # # ... have any information inside the query, thus we have to count the term frequency of the query with respect
        # # ... to the current term
        # f = self.get_frequency_from_doc(normalized_term, doc_name, directory_path)
        # if f > 0:
        #     return 1 + math.log(f)

        return 0

    def idf(self, normalized_term):
        return 1 + math.log(
            float(len(self.doc_class_frequency) + 1) / float(len(self.word_doc_frequency[normalized_term]) + 1))

    def tf_idf(self, normalized_term, doc_name, directory_path=""):
        return self.tf(normalized_term, doc_name, directory_path) * self.idf(normalized_term)

    def similarity(self, query, doc_name, directory_path_1="", directory_path_2=""):
        # import numpy as np
        # from numpy import linalg as LA

        # v1 = np.array([])
        # v2 = np.array([])

        dot_product = 0.0
        d1_square_norm = 0.0
        d2_square_norm = 0.0

        freq_table = self.get_table_frequencies_from_doc(query, directory_path_1)

        for t in self.word_doc_frequency.keys():
            # v1 = np.append(v1, self.tf_idf(t, doc_1_name, directory_path_1))
            # v2 = np.append(v2, self.tf_idf(t, doc_2_name, directory_path_2))
            v1 = (1 + math.log(freq_table[t]) if freq_table[t] > 0 else 0) * self.idf(t)
            v2 = self.tf_idf(t, doc_name, directory_path_2)
            dot_product += v1 * v2

            d1_square_norm += v1 * v1
            d2_square_norm += v2 * v2

        return float(dot_product) / (float(math.sqrt(d1_square_norm)) * float(math.sqrt(d2_square_norm)))
        # return np.dot(v1, v2) / (LA.norm(v1) * LA.norm(v2))






