from sklearn.model_selection import train_test_split
from random import randint
import string
from nltk.tokenize import ToktokTokenizer 

def load_parallel(path, prefix, l1, l2):
    l1_path = path+'/'+prefix+'.'+l1
    l2_path = path+'/'+prefix+'.'+l2

    l1_all_sentences = []
    l2_all_sentences = []

    c1 = 0
    c2 = 0

    with open(l1_path, mode='rt', encoding='utf-8') as l1_file:
        for line in l1_file:
            if line != '\n':
                l1_all_sentences.append(line.strip())
            else:
                c1 += 1
                l1_all_sentences.append('NULL')

            if len(line.split(' ')) == 0:
                print('oh noes')

    with open(l2_path, mode='rt', encoding='utf-8') as l2_file:
        for line in l2_file:
            if line != '\n':
                l2_all_sentences.append(line.strip())
            else:
                c2 += 1
                l2_all_sentences.append('NULL')

    print(c1, c2)

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
    ret = ' '.join([word for word in list_of_words]).strip()
    if ret == '':
        ret = 'null'
    return ret

def stripper_detoken(list_of_sentences):
    tokenized_sentences = []

    for sentence in list_of_sentences:
        tokenized_sentences.append(sentence_from_tokens(sentence_tokenizer(sentence)))

    return tokenized_sentences
    

import json
def load_europarl(directory, jsonmode=True):
    print("Loading all sentences")
    en_all_sentences, fr_all_sentences = load_parallel(directory, 'europarl-v7.fr-en', 'en', 'fr') 
    print("Done")

    if jsonmode:
        directory = directory + '/json/'
    else:
        directory = directory + '/txt/'

    print("Stripping sentences of punctuation")
    en_all_sentences = stripper_detoken(en_all_sentences)
    fr_all_sentences = stripper_detoken(fr_all_sentences)
    print("Done")

    print("Splitting into sets")
    en_train, en_test, en_dev, fr_train, fr_test, fr_dev = split_train_test_dev(en_all_sentences, fr_all_sentences)
    print("Done")

    if jsonmode:
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
    else:
        print('Writing the train files')
        with open(directory+'en_train.txt', 'w') as fileout:
            fileout.write(strfy(en_train))
        with open(directory+'fr_train.txt', 'w') as fileout:
            fileout.write(strfy(fr_train))

        print('Writing the dev files')
        with open(directory+'en_dev.txt', 'w') as fileout:
            fileout.write(strfy(en_dev))
        with open(directory+'fr_dev.txt', 'w') as fileout:
            fileout.write(strfy(fr_dev))

        print('Writing the test files')
        with open(directory+'en_test.txt', 'w') as fileout:
            fileout.write(strfy(en_test))
        with open(directory+'fr_test.txt', 'w') as fileout:
            fileout.write(strfy(fr_test))

def strfy(stringlist):
    newstr = ''
    for line in stringlist:
        newstr += line
        newstr += '\n'
    return newstr


import sys
if __name__ == "main":
    if len(sys.argv) < 2:
        print("Execution: python {0} <filename>".format(sys.argv[0], jsonmode=False))
    else:
        load_europarl(sys.argv[1])








