from sklearn.model_selection import train_test_split
from random import randint
import string
from nltk.tokenize import ToktokTokenizer 

def load_parallel(path, prefix, l1, l2):
    l1_path = path+'/'+prefix+'.'+l1
    l2_path = path+'/'+prefix+'.'+l2

    l1_all_sentences = []
    l2_all_sentences = []

    with open(l1_path, mode='rt', encoding='utf-8') as l1_file:
        for line in l1_file:
            l1_all_sentences.append(line.strip())

    with open(l2_path, mode='rt', encoding='utf-8') as l2_file:
        for line in l2_file:
            l2_all_sentences.append(line.strip())

    return l1_all_sentences, l2_all_sentences

def split_train_test_dev(lang1_sentences, lang2_sentences):

    train_splits = 0.8
    test_dev_splits = 0.5

    l1_train, l1_test, l2_train, l2_test = train_test_split(
            lang1_sentences,
            lang2_sentences,
            train_size = train_splits
        ) 

    l1_test, l1_dev, l2_test, l2_dev = train_test_split(
            l1_test,
            l2_test,
            train_size = test_dev_splits
        )

    return l1_train, l1_test, l1_dev, l2_train, l2_test, l2_dev
    
def sentence_tokenizer(sentence):
    toktok = ToktokTokenizer()
    return [i.lower() for i in toktok.tokenize(sentence) if i not in string.punctuation]

def sentence_from_tokens(list_of_words):
    return ' '.join([word for word in list_of_words]).strip()

def stripper_detoken(list_of_sentences):
    tokenized_sentences = []

    for sentence in list_of_sentences:
        tokenized_sentences.append(sentence_from_tokens(sentence_tokenizer(sentence)))

    return tokenized_sentences
    

import json
def load_europarl(directory):
    print("Loading all sentences")
    en_all_sentences, fr_all_sentences = load_parallel(directory, 'europarl-v7.fr-en', 'en', 'fr') 
    print("Done")

    directory = directory + '/json/'

    print("Stripping sentences of punctuation")
    en_all_sentences = stripper_detoken(en_all_sentences)
    fr_all_sentences = stripper_detoken(fr_all_sentences)
    print("Done")

    print("Splitting into sets")
    en_train, en_test, en_dev, fr_train, fr_test, fr_dev = split_train_test_dev(en_all_sentences, fr_all_sentences)
    print("Done")

    print("Writing train to train.json")
    # Yes, this is slow because I'm opening a new file object for every single 
    # line, but it was a rushed job okay
    for i in range(len(en_train)):
        line = {}
        line['src'] = en_train[i]
        line['trg'] = fr_train[i]

        with open(directory+'train.json', 'a+') as json_out:
            json.dump(line, json_out, ensure_ascii=False)
            json_out.write('\n')
    print("Done")

    print("Writing test to test.json")
    for i in range(len(en_test)):
        line = {}
        line['src'] = en_test[i]
        line['trg'] = fr_test[i]

        with open(directory+'test.json', 'a+') as json_out:
            json.dump(line, json_out, ensure_ascii=False)
            json_out.write('\n')
    print("Done")

    print("Writing dev to dev.json")
    for i in range(len(en_dev)):
        line = {}
        line['src'] = en_dev[i]
        line['trg'] = fr_dev[i]

        with open(directory+'dev.json', 'a+') as json_out:
            json.dump(line, json_out, ensure_ascii=False)
            json_out.write('\n')
    print("Done")

import sys
if __name__ == "main":
    if len(sys.argv) < 2:
        print("Execution: python {0} <filename>".format(sys.argv[0]))
    else:
        load_europarl(sys.argv[1])








