'''
    In this script we define some utility functions going to use for read and pre-process data.
    M. Jafari August 1st, 2018
'''

from re import compile, sub

from logging import info, basicConfig, INFO
from json import dumps, loads
from random import randint
import matplotlib.pyplot as plt
import time


basicConfig(level=INFO)

def read_data_to_dict(raw_file_path, clean_file, dictionary, minimum_count, maximum_count):
    word_dictionary = {}
    id = 0
    with open(raw_file_path, 'r', errors='ignore') as raw_file:
    # with open(clean_file, 'w', errors='ignore') as clean_file:
        for line in raw_file:
            for word in line.split():
                if word.isalpha():
                    word = word.lower()
                    if word not in word_dictionary:
                        word_dictionary[word] = [id, 1]
                        id = id + 1
                        # clean_file.write(','+str(id))
                        if id%1000 == 0:
                            info("%d unique words fined till now!"%id)
                    else:
                        word_dictionary[word][1] = word_dictionary[word][1]+1
                        # clean_file.write(',' + str(word_dictionary[word][0]))
    raw_file.close()
    temporary_dict = {}
    for key in word_dictionary:
        if ~(word_dictionary[key][1]> maximum_count or word_dictionary[key][1]<minimum_count):
            temporary_dict[key] = word_dictionary[key]
    word_dictionary = temporary_dict
    counter = 0
    with open(raw_file_path, 'r', errors='ignore') as raw_file:
        with open(clean_file, 'w', errors='ignore') as clean_file:
            for line in raw_file:
                for word in line.split():
                    if word.isalpha():
                        word = word.lower()
                        if word in word_dictionary:
                            clean_file.write(',' + str(word_dictionary[word][0]))
                            counter = counter + 1
                            if counter%10000 == 0:
                                info("%dth word is written in clear text file!"%(counter))
    with open(dictionary, 'w') as dict_file:
        dict_file.write(dumps(word_dictionary))
    del word_dictionary


def batch_generator(batch_size, window_size, negative_sample_count,
                    clear_text):
    with open(clear_text, 'r') as file:
        content = file.read().split(',')
        document_length = len(content)-2
        train_data = content[1:-1]
        del content

    while True:
        batch = []
        for i in range(0, batch_size):
            # use uniform distribution
            center_word = randint(window_size//2, document_length-window_size//2)
            sample = [int(train_data[center_word])+1]
            # sample = sample + [-1]
            for j in range(1, window_size//2+1):
                sample = sample + [int(train_data[center_word-j])+1]
                break
                # sample = sample + [int(train_data[center_word+j])]
            # sample = sample + [-1]
            # for j in range(0, negative_sample_count):
            #     random_sample = randint(0, document_length)
            #     if train_data[random_sample] not in sample:
            #         sample = sample+[int(train_data[random_sample])]
            batch.append(sample)

        yield batch






def histogram(count_dictionary):
    with open(count_dictionary, 'r') as dict:
        my_dict = dict.read()
        my_dict = loads(my_dict)
    counts = sorted([my_dict[x][1] for x in my_dict if (my_dict[x][1]>5 and my_dict[x][1]<2000) ])
    print(counts)
    plt.hist(counts)

    plt.show()



if __name__ == "__main__":
    # path = "C:\\Users\\METI\\Desktop\\sample.txt"
    path1 = "G:\\desktop\\quora - Copy.tsv"
    path2 = "clean_file.txt"
    path3 = "dictionary"

    # read_data_to_dict(path1, path2, path3, minimum_count=5, maximum_count=2000)
    #
    start_time = time.time()
    # your code

    batch = batch_generator(batch_size=10, window_size=5, negative_sample_count=10,
                            clear_text= path2)

