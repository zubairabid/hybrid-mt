#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

import pickle


# In[2]:


def load_stopwords(path_to_stopwords):
    stopwords = []
    
    with open(path_to_stopwords, 'rb') as f:
        stopwords = pickle.load(f)
    
    return stopwords


# In[3]:


def load_index_from_word(path_to_en2id):
    en2id = {}
    
    with open(path_to_en2id, 'rb') as f:
        en2id = pickle.load(f)
        
    return en2id


# In[4]:


def load_lookup_table(path_to_lookup_table):
    lookup_table = []

    with open(path_to_lookup_table, 'rb') as f:
        lookup_table = pickle.load(f)
    
    return lookup_table


# In[5]:


def preprocess(sentences):
    processed_sentences = []
    
    for sentence in sentences:
        processed_sentences.append(sentence.lower())
        
    return processed_sentences


# In[6]:


def topics_from_dataset(sentences):   
    print("Generating topics and weights for dataset")
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(sentences))
    topics = vectorizer.get_feature_names() 
    weights = tfidf.toarray()
    
    return topics, weights


# In[7]:


def sentence_remove_stopwords(sentence, stopwords):
    filtered_words = []
    reduced_sentence = ''
    
    wordlist = sentence.strip().split(' ')
    for word in wordlist:
        if word not in stopwords:
            filtered_words.append(word)
    reduced_sentence = ' '.join(filtered_words)
    
    return reduced_sentence


# In[8]:


def topics_from_sentence(sentence_id, sentence, weights, topics):   
    top_topics = []
    sentence_topics = []
    weight = weights[sentence_id]
    location = np.argsort(-weight)

    limit = min(10, len(weight))

    for i in range(limit):
        if weight[location[i]] > 0.0:
            top_topics.append(topics[location[i]])
    
    for word in sentence.split():
        if word.lower() in top_topics:
            sentence_topics.append(word)
    
    return sentence_topics


# In[9]:


def images_from_topics(sentence_topics, stopwords, en2id, lookup_table):
    
    imagelist = []
    
    for topic in sentence_topics:
        if topic in en2id.keys() and not topic in stopwords:
            if en2id[topic] in lookup_table:
                #print('<', topic, '> is in lookup table')
                #print(topic, lookup_table[en2id[topic]])
                for image in lookup_table[en2id[topic]]:
                    if image > 0.0 and not image in imagelist:
                        imagelist.append(image)
            else:
                pass
                #print('>', topic, '< not in lookup table')
        else:
            if topic not in en2id.keys():
                pass
                #print('|', topic, '| not in dictionary')
                
    return imagelist


# In[10]:


def get_features(sentences, cap):
    path_to_en2id = 'en2id.pkl'
    path_to_stopwords = 'stopwords-en.pkl'
    path_to_lookup_table = 'cap2image_en2fr.pickle'

    sentences = preprocess(sentences)
    images_for_sentence = []
    en2id = load_index_from_word(path_to_en2id)
    stopwords = load_stopwords(path_to_stopwords)
    lookup_table = load_lookup_table(path_to_lookup_table)

    topics, weights = topics_from_dataset(sentences)

    for sentence_id, sentence in enumerate(sentences):
        sentence_topics = topics_from_sentence(sentence_id, sentence, weights, topics)
        imagelist = images_from_topics(sentence_topics, stopwords, en2id, lookup_table)
        if not imagelist:
            imagelist=[0]
        images_for_sentence.append(imagelist)
    
    feature_index = np.load('./data/train-resnet50-avgpool.npy')
    batch_sentence_features = []
    for i, dummy in enumerate(sentences):
        sentence = sentences[i]
        images = images_for_sentence[i]
        sentence_features = []
        for image in images:
            image_feature = feature_index[image-1]
            sentence_features.append(image_feature)
        if len(sentence_features) > cap:
            sentence_features = sentence_features[:cap]
        elif len(sentence_features) < cap:
            for j in range(cap-len(sentence_features)):
                sentence_features.append(np.zeros((2048,), dtype=float ))
        batch_sentence_features.append(sentence_features)
    
    pt = np.array(batch_sentence_features)
    return pt


