__author__ = 'macpro'
import re
import os
import collections
from PorterStemmer import PorterStemmer
import math
from Queue import Queue
import linecache
import glob
from PriorityQueue import PriorityQueue


def read_words(file_name):
    last = ""
    with open(file_name) as inp:
        while True:
            buf = inp.read(10240)
            if not buf:
                break
            words = (last+buf).split()
            last = words.pop()
            for word in words:
                yield word
        yield last


class MyIndexer:
    class Block:
        def __init__(self, file_sorted, cur_seek, limit):
            self.cur_seek = cur_seek
            self.limit = limit
            self.data_file = file_sorted
            self.BUFFER_SIZE = 20

        def get_next_buffer(self):
            result = Queue()
            n = self.BUFFER_SIZE + self.cur_seek
            if n > self.limit:
                return result

            for i in range(self.cur_seek, n):
                temp = linecache.getline(self.data_file, i).split()
                result.put(temp)
            self.cur_seek += self.BUFFER_SIZE
            return result

        def is_empty(self):
            if self.cur_seek > self.limit:
                return True
            return False


    def __init__(self, stop_words_file=""):
        self.stop_words = collections.defaultdict(lambda: False)
        self.stemmer = PorterStemmer()
        self.map_trans_terms = collections.defaultdict(lambda: 0)
        self.map_trans_docs = collections.defaultdict(lambda: 0)
        self.total_words = 0

        if stop_words_file != "":
            with open(stop_words_file, 'r') as f:
                for line in f:
                    for w in line.split():
                        w = self.normalize(w)
                        self.stop_words[w] = True

        self.BLOCK_SIZE = 100 # Number of entity, each record is a (term, docID)

    def get_doc_class_frequency(self):
        return self.doc_class_frequency

    def get_stop_words(self):
        return self.stop_words

    def get_word_doc_frequency(self):
        return self.word_doc_frequency

    def parse(self, data_directory):
        files_name = os.listdir(data_directory)
        if files_name.count(".DS_Store") > 0:
            files_name.remove(".DS_Store")

        cur_block_size = 0

        LAST_ID = 1
        LAST_FILE_ID = 0

        records_list = []
        write_type = 'wb+'
        count = 0

        exist = collections.defaultdict(lambda: collections.defaultdict(lambda: False))

        for f_name in files_name:
            LAST_FILE_ID += 1
            self.map_trans_docs[f_name] = LAST_FILE_ID
            cur_file_id = LAST_FILE_ID

            x = (float(count) / float(len(files_name))) * 100
            print "Parsing : " + str(x) + " %"
            count += 1

            for word in read_words(data_directory + "/" + f_name):
                word = self.normalize(word)
                if self.stop_words[word] > 0 or len(word) == 0:
                    continue

                # I do not use word because the word length could be 15, but my string base is 37, thus, we may have
                # ... 37^15 => I cant not fix it. So, I use word term instead, because I find no way to write int in python,
                # ... so long int will become long string and it will be worse than using term as its own ID

                # if self.map_trans_terms[word] == 0:
                #     self.map_trans_terms[word] = LAST_ID
                #     LAST_ID += 1
                # cur_term_id = self.map_trans_terms[word]

                records_list.append((word, cur_file_id))

                # Each int in python account for 24 bytes => 24 bytes for termID and 24 bytes for docID
                cur_block_size += 1

                if cur_block_size == self.BLOCK_SIZE:
                    with open("file_raw", write_type) as fr:
                        for record in records_list:
                            if exist[record[0]][record[1]] == False:
                                exist[record[0]][record[1]] = True

                                fr.write(bytes(record[0]))
                                fr.write(bytes(" "))
                                fr.write(bytes(record[1]))
                                fr.write(bytes("\n"))
                                self.total_words += 1
                        write_type = 'ab+'

                    records_list = []
                    cur_block_size = 0


            if cur_block_size > 0:
                with open("file_raw", write_type) as fr:
                    for record in records_list:
                        if exist[record[0]][record[1]] == False:
                            exist[record[0]][record[1]] = True

                            fr.write(bytes(record[0]))
                            fr.write(bytes(" "))
                            fr.write(bytes(record[1]))
                            fr.write(bytes("\n"))
                            self.total_words += 1
                    write_type = 'ab+'

        print "OK!"
        print "----------------------------------------------------"

        print "Writing map docs file ... "
        with open("doc_docID", 'w') as f:
            for f_name in self.map_trans_docs.keys():
                f.write(str(self.map_trans_docs[f_name]) + " ")
                f.write(f_name + "\n")
        print "OK!"
        print "----------------------------------------------------"

    def sort(self, file_raw):
        records = []

        cur_block_size = 0
        write_type = 'wb+'
        count = 0

        with open(file_raw, 'r') as f:
            for line in f:

                x = (float(count) / float(self.total_words)) * 100
                print "Sorting : " + str(x) + " %"
                count += 1

                temp = line.split()
                records.append((temp[0], temp[1]))

                # Each int in python account for 24 bytes => 24 bytes for termID and 24 bytes for docID
                # And we assume that the block size is calculated on number of record (term, docID)
                cur_block_size += 1

                if cur_block_size == self.BLOCK_SIZE:
                    with open("file_sorted", write_type) as fr:

                        for record in sorted(records, key=lambda x: (x[0], x[1]), reverse=False):
                            fr.write(bytes(record[0]))
                            fr.write(bytes(" "))
                            fr.write(bytes(record[1]))
                            fr.write(bytes("\n"))
                        write_type = 'ab+'
                    records = []
                    cur_block_size = 0

            if cur_block_size > 0:
                with open("file_sorted", write_type) as fr:
                    for record in sorted(records, key=lambda x: (x[0], x[1]), reverse=False):
                        fr.write(bytes(record[0]))
                        fr.write(bytes(" "))
                        fr.write(bytes(record[1]))
                        fr.write(bytes("\n"))

        print "OK!"
        print "----------------------------------------------------"

    def merge_buffers(self, buff1, buff2):
        r_buffer = Queue()

        if buff1.empty() and buff2.empty():
            return buffer

        if buff1.empty():
            return buff2
        if buff2.empty():
            return buff1

        temp1 = buff1.get()
        temp2 = buff2.get()

        while True:
            # if waiting != None:
            #     t = [temp1, temp2, waiting]
            #     cur_max = [k for k in t if k[0] == max(y[0] for y in t)]
            #     c = set(t) - {cur_max}
            #     c = list(c)
            #     temp1 = c[0]
            #     temp2 = c[1]
            #     waiting = cur_max

            if temp1[0] == temp2[0]:
                temp_posting_list = [temp1[0]]
                temp1.pop(0)
                temp2.pop(0)
                a = set(temp1)
                b = set(temp2)
                merge_list = [x for x in a.union(b)]
                temp_posting_list.extend(merge_list)
                r_buffer.put(temp_posting_list)

                if buff1.empty() or buff2.empty():
                    break

                temp1 = buff1.get()
                temp2 = buff2.get()

            elif temp1[0] < temp2[0]:
                r_buffer.put(temp1)
                if buff1.empty():
                    break

                temp1 = buff1.get()
            else:
                r_buffer.put(temp2)
                if buff2.empty():
                    break

                temp2 = buff2.get()

        # Post processing for the remaining records

        while not buff1.empty():
            r_buffer.put(buff1.get())

        while not buff2.empty():
            r_buffer.put(buff2.get())

        return r_buffer

    def read_buffer(self, buff_file):
        buff = Queue()
        with open(buff_file, 'r') as f:
            for line in f:
                temp = line.split()
                buff.put(temp)
        return buff

    def write_buffer(self, buff, file_name, t):
        with open(file_name, t) as f:
            while not buff.empty():
                temp = buff.get()
                for t in temp:
                    f.write(bytes(t))
                    f.write(bytes(" "))
                f.write(bytes("\n"))

    def merge(self, file_sorted):
        temp_dir = "temporary_data "
        block_merged_file = "merged_"
        NUM_BLOCK = int(math.ceil(float(self.total_words) / float(self.BLOCK_SIZE)))

        # Pre-process
        source_buffers = [self.Block(file_sorted, x * self.BLOCK_SIZE + 1, (x + 1) * self.BLOCK_SIZE) for x in range(0, NUM_BLOCK)]

        # The limit of the last buffer should be fit with the data set
        source_buffers[len(source_buffers) - 1].limit = self.total_words

        for i in range(0, len(source_buffers) - 1, 2):

            print "Merging Phase 1 : " + str((float(i) / (float(len(source_buffers) - 1))) * 100) + " %"
            t = 'wb+'
            while not source_buffers[i].is_empty() or not source_buffers[i+1].is_empty():
                buff1 = source_buffers[i].get_next_buffer()
                buff2 = source_buffers[i+1].get_next_buffer()
                merge_buff = self.merge_buffers(buff1, buff2)
                self.write_buffer(merge_buff, temp_dir + "/" + block_merged_file + str(i), t)
                t = 'ab+'

        if i == len(source_buffers) - 1:
            t = 'wb+'
            while not source_buffers[i].is_empty():
                buff1 = source_buffers[i].get_next_buffer()
                self.write_buffer(buff1, temp_dir + "/" + block_merged_file + str(i), t)
                t = 'ab+'

        dirs = os.listdir(temp_dir)

        n = len(dirs)

        while len(dirs) > 1:
            print "Merging Phase 2 : " + str((1 - float(len(dirs)) / float(n)) * 100) + " %"
            # We do not need to modify the limit of the last block because linecache will return "" for the line of file
            for i in range(0, len(dirs) - 1, 2):
                buff1 = self.read_buffer(temp_dir + "/" + dirs[i])
                buff2 = self.read_buffer(temp_dir + "/" + dirs[i+1])

                merge_buff = self.merge_buffers(buff1, buff2)
                self.write_buffer(merge_buff, temp_dir + "/" + dirs[i])

                # The next file become redundant
                os.unlink(temp_dir + "/" + dirs[i+1])

            if i == len(source_buffers) - 1:
                buff1 = source_buffers[i].get_next_buffer()
                self.write_buffer(buff1, temp_dir + "/" + block_merged_file + str(i))

            dirs = os.listdir(temp_dir)

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
        ws = re.findall(r'[a-zA-Z0-9]+-?[a-zA-Z0-9]+', w, re.VERBOSE)

        if len(ws) > 0:
            return ws[0]
        return ""

    def get_frequency_from_doc(self, normalized_term, doc_name, directory_path):
        count_frequency = 0
        with open(directory_path + "/" + doc_name, 'r') as f:
            words = re.findall(r'[a-zA-Z0-9]+-?[a-zA-Z0-9]+', f.read(), re.VERBOSE)
            for w in words:
                w = self.normalize(w)
                if w == normalized_term and self.stop_words[w] == 0:
                    count_frequency += 1

        return count_frequency

    def tf(self, normalized_term, doc_name, directory_path=""):
        if self.stop_words[normalized_term] > 0:
            return 0

        f = self.word_doc_frequency[normalized_term][doc_name]
        if f > 0:
            return 1 + math.log(f)

        # This is the case when doc is query. Because we did not include the query in the train set, thus we do not
        # ... have any information inside the query, thus we have to count the term frequency of the query with respect
        # ... to the current term
        f = self.get_frequency_from_doc(normalized_term, doc_name, directory_path)
        if f > 0:
            return 1 + math.log(f)

        return 0

    def idf(self, normalized_term):
        return 1 + math.log(float(len(self.doc_class_frequency)) / float(len(self.word_doc_frequency[normalized_term])))

    def tf_idf(self, term, doc_name, directory_path=""):
        term = self.normalize(term)
        return self.tf(term, doc_name, directory_path) * self.idf(term)

    def similarity(self, doc_1_name, doc_2_name, directory_path_1="", directory_path_2=""):
        import numpy as np
        from numpy import linalg as LA

        # v1 = np.array([])
        # v2 = np.array([])

        dot_product = 0.0
        d1_square_norm = 0.0
        d2_square_norm = 0.0

        for t in sorted(self.word_doc_frequency.iterkeys()):
            # v1 = np.append(v1, self.tf_idf(t, doc_1_name, directory_path_1))
            # v2 = np.append(v2, self.tf_idf(t, doc_2_name, directory_path_2))
            v1 = self.tf_idf(t, doc_1_name, directory_path_1)
            v2 = self.tf_idf(t, doc_2_name, directory_path_2)
            dot_product += v1 * v2
            d1_square_norm += v1 * v1
            d2_square_norm += v2 * v2

        return float(dot_product) / (float(math.sqrt(d1_square_norm)) * float(math.sqrt(d2_square_norm)))
        # return np.dot(v1, v2) / (LA.norm(v1) * LA.norm(v2))






