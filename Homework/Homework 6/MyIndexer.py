__author__ = 'macpro'
import re
import os
import collections
from PorterStemmer import PorterStemmer
import math

WORD_PATTERN = r'\b[a-zA-Z]+-?[a-zA-Z]+|\b[1-9]{1,3}\.?[0-9]{1,3}'
MAX_WORDS_PER_CLASS = 8000
STEP = 1
K = 50

class MyIndexer:
    def __init__(self, stop_words_file=""):
        self.word_doc_frequency = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        self.doc_class_frequency = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        self.total_words = 0

        self.class_word_frequency = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        self.class_doc_frequency = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))

        self.stop_words = collections.defaultdict(lambda: False)
        self.stemmer = PorterStemmer()

        self.test_set = collections.defaultdict(lambda: [])
        self.total_doc_test_set = 0

        self.train_set = collections.defaultdict(lambda: [])
        self.total_doc_train_set = 0

        self.data_set_directory = ""

        if stop_words_file != "":
            with open(stop_words_file, 'r') as f:
                for line in f:
                    for w in line.split():
                        w = self.normalize(w)
                        self.stop_words[w] = True

    def get_test_set(self):
        return self.test_set

    def get_doc_class_frequency(self):
        return self.doc_class_frequency

    def get_stop_words(self):
        return self.stop_words

    def get_word_doc_frequency(self):
        return self.word_doc_frequency

    def split_train_test_sets(self, data_directory):
        dirs = os.listdir(data_directory)
        if dirs.count(".DS_Store") > 0:
            dirs.remove(".DS_Store")

        self.data_set_directory = data_directory

        for subdir_name in dirs:
            path = data_directory + '/' + subdir_name
            files_name = os.listdir(path)
            if files_name.count(".DS_Store") > 0:
                files_name.remove(".DS_Store")

            count_file = 0
            for f_name in files_name:
                count_file += 1

                # Choose 5000 documents for testing (5000 / 20 = 250, each class has 1000 document, thus 1000 / 250 = 4)
                if count_file % 4 == 0:
                    (self.test_set[subdir_name]).append(f_name)
                    self.total_doc_test_set += 1
                    continue

                self.doc_class_frequency[f_name][subdir_name] += 1
                self.class_doc_frequency[subdir_name][f_name] += 1

                (self.train_set[subdir_name]).append(f_name)
                self.total_doc_train_set += 1


    def index_data(self):
        count_words_class = 0
        count = 1
        count_file = 0
        n = len(self.train_set)
        for c in self.train_set.keys():
            print "Reading progress : " + str(count) + "/" + str(n) + " directories...",
            for d in self.train_set[c]:
                count_file += 1

                # We just pick 2000 terms, thus each class has 2000/20 = 100 terms, thus each file has 100 / 1000 ...
                # ... => for every STEP files, we have 1 file used to add to dict
                if count_file % STEP != 0:
                    continue

                temp = self.read_file(self.data_set_directory + '/' + c + "/" + d, count_words_class, c + "_" + d, MAX_WORDS_PER_CLASS)
                count_words_class = temp[1]
                self.total_words += temp[1]

                if temp[0] is False:
                    break

            print " OK!"
            count += 1

    def read_file(self, file_path, count_words_class, doc_name, MAX_WORDS_PER_CLASS):
        with open(file_path, 'r') as f:
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
                        if self.word_doc_frequency[w][doc_name] == 0:
                            count_words_class += 1
                            c = (doc_name.split("_"))[0]
                            self.class_word_frequency[c][w] += 1

                        self.word_doc_frequency[w][doc_name] += 1
                        self.total_words += 1

                        if count_words_class > MAX_WORDS_PER_CLASS:
                            return False, count_words_class
        return True, count_words_class

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

    def tf(self, normalized_term, doc_name):
        if self.stop_words[normalized_term] > 0:
            return 0

        f = self.word_doc_frequency[normalized_term][doc_name]
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

    def tf_idf(self, normalized_term, doc_name):
        return self.tf(normalized_term, doc_name) * self.idf(normalized_term)

    def similarity(self, freq_table, doc_name):
        # import numpy as np
        # from numpy import linalg as LA

        # v1 = np.array([])
        # v2 = np.array([])

        dot_product = 0.0
        d1_square_norm = 0.0
        d2_square_norm = 0.0

        for t in self.word_doc_frequency.keys():
            # v1 = np.append(v1, self.tf_idf(t, doc_1_name, directory_path_1))
            # v2 = np.append(v2, self.tf_idf(t, doc_2_name, directory_path_2))
            v1 = (1 + math.log(freq_table[t]) if freq_table[t] > 0 else 0) * self.idf(t)
            v2 = self.tf_idf(t, doc_name)
            dot_product += v1 * v2

            d1_square_norm += v1 * v1
            d2_square_norm += v2 * v2

        temp = (float(math.sqrt(d1_square_norm)) * float(math.sqrt(d2_square_norm)))
        if temp > 0:
            return float(dot_product) / (float(math.sqrt(d1_square_norm)) * float(math.sqrt(d2_square_norm)))
        return 0
        # return np.dot(v1, v2) / (LA.norm(v1) * LA.norm(v2))

    def get_words_from_file(self, file_path):
        result = []
        with open(file_path, 'r') as f:
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
                        result.append(w)

        return result

    def classify_k_nearest(self, freq_table, testing_count):
        import operator
        scores = collections.defaultdict(lambda: 0)

        count = 0
        for c in self.train_set.keys():
            for d in self.train_set[c]:
                count += 1
                doc_id = c + "_" + d
                scores[doc_id] = self.similarity(freq_table, doc_id)
        result = sorted(scores.iteritems(), key=operator.itemgetter(1), reverse=True)[:K-1]

        final_class = None
        max_k = 0
        for r in result:
            c = r.split("_")[0]
            k = self.class_doc_frequency[c].keys()
            if k > max_k:
                max_k = k
                final_class = c
        return final_class

    def classify_naive_bayes(self, words, class_total_words):
        import operator
        score_per_class = collections.defaultdict(lambda: 0)
        V = len(self.word_doc_frequency)

        for w in words:
            for c in self.class_word_frequency.keys():
                score_per_class[c] += math.log(self.class_word_frequency[c][w] + 1)
                score_per_class[c] -= math.log(class_total_words[c] + V)

        return max(score_per_class.iteritems(), key=operator.itemgetter(1))[0]

    def test_classifier_naive_bayes(self):
        count_correct = 0

        count_progress = 0
        n = len(self.test_set)

        class_total_words = collections.defaultdict(lambda: 0)
        for c in self.class_word_frequency.keys():
            class_total_words[c] = sum([self.class_word_frequency[c][x] for x in self.class_word_frequency[c].keys()])

        for c in self.test_set.keys():
            count_progress += 1
            print "Progress : " + str((float(count_progress) / float(n)) * 100) + " %"
            for d in self.test_set[c]:
                words = self.get_words_from_file(self.data_set_directory + "/" + c + "/" + d)
                if len(words) > 0:
                    predict_c = self.classify_naive_bayes(words, class_total_words)
                    if predict_c == c:
                        count_correct += 1

        return float(count_correct) / float(self.total_doc_test_set)

    def test_classifier_k_nearest(self):
        count_correct = 0

        count_progress = 0
        n = len(self.test_set)

        class_total_words = collections.defaultdict(lambda: 0)
        for c in self.class_word_frequency.keys():
            class_total_words[c] = sum([self.class_word_frequency[c][x] for x in self.class_word_frequency[c].keys()])

        for c in self.test_set.keys():
            for d in self.test_set[c]:
                freq_table = self.get_table_frequencies_from_doc(d, self.data_set_directory + "/" + c)
                if len(freq_table) > 0:
                    predict_c = self.classify_k_nearest(freq_table)
                    if predict_c == c:
                        count_correct += 1

        return float(count_correct) / float(self.total_doc_test_set)






