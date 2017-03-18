__author__ = 'Tran Tinh Chi'

from MyIndexer import MyIndexer

def main():

    data_directory = "data"
    stop_words = "stopwords_en.txt"

    indexer = MyIndexer(stop_words)
    indexer.parse(data_directory)
    indexer.sort("file_raw")
    indexer.merge("file_sorted")

if __name__ == "__main__":
    main()