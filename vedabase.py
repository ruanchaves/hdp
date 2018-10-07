# HDP algorithm applied to the Vedabase.
# It took it about 54 minutes to process the entire Vedabase on a Ryzen 7
# with 16GB of RAM.

# Folder where the text files for the database are stored.
db_path = './txt/'

# Text files format.
db_format = 'txt'

# Ignore the 10% most frequent words.
THRESHOLD = 0.01

# Token length.
TOKEN_LENGTH = 1000

# This file stores the output in a format suitable for being the input of
# an Elasticsearch database.
input_file = 'inputfile.json'

# This file stores the output as a human-readable json list of dictionaries.
vedabase_corpus = 'vedabasecorpus.json'

# Find the 200 most similar entries to every entry on the database.
similar_size = 200

# Approximate nearest neighbors tree depth. Smaller values mean faster
# execution and less memory usage.
ann_tree_depth = 2000

# Elasticsearch index and type
elastic_index = 'vedabase'
elastic_type = 'tokens'

# Translate broken Balarama font diacritics into plain ASCII characters.
diact_dct = {
    'ù': 'h',
    'Ù': 'H',

    'ï': 'n',
    'Ï': 'N',

    'ç': 's',
    'Ç': 'S',

    'ë': 'n',
    'Ë': 'N',

    'é': 'i',
    'É': 'I',

    'ò': 'd',
    'Ö': 'T',

    'å': 'r',
    'Å': 'R',

    'ö': 'h',
    'Ö': 'H',

    'ä': 'a',
    'Ä': 'A',

    'ñ': 's',
    'Ñ': 'S',

    'à': 'm',
    'À': 'M',

    'ü': 'u',
    'Ü': 'U'
}

translator = "".maketrans(diact_dct)

import logging

# logging

logger = logging.getLogger('ElasticHDP')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('elastichdp.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

logger.info('Loading imports.')

# imports

import os
import subprocess
import pandas as pd
import numpy as np
import random
import re
import string
import json

import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist

from gensim import corpora
from gensim.models.hdpmodel import HdpModel

from annoy import AnnoyIndex

from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch

# preprocessing
logger.info('Starting preprocessing.')


def preprocessing():
    filelist = list(map((lambda x: db_path + x), os.listdir(db_path)))
    files = list(filter((lambda x: x),
                        [x if db_format in x[-4:] else None for x in filelist]))

    documents = []

    for line, fname in enumerate(files):
        with open(fname, 'r') as f:
            txt = f.read().split('\n')

        file_progress = "{:.2%}".format(line / len(files) )
        # translate
        logger.info(fname + '\n Translating. ' + file_progress)
        txt = [line.translate(translator) for line in txt]

        # backup
        logger.info(fname + '\n Backing up. ' + file_progress)
        original_txt = txt[::]

        # tokenize
        logger.info(fname + '\n Tokenizing. ' + file_progress)
        txt = [word_tokenize(line) for line in txt]

        # remove punctuation
        logger.info(fname + '\n Removing punctuation. ' + file_progress)
        pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
        for i in range(len(txt)):
            txt[i] = list(filter(None, [pattern.sub('', token)
                                        for token in txt[i]]))

        # lowercase
        logger.info(fname + '\n Lowercasing. ' + file_progress)
        for i in range(len(txt)):
            txt[i] = [x.lower() for x in txt[i]]

        # remove stopwords
        logger.info(fname + '\n Removing stopwords. ' + file_progress)
        sw = nltk.corpus.stopwords.words('english')
        for i in range(len(txt)):
            txt[i] = [w for w in txt[i] if w not in sw]

        # stemmer
        logger.info(fname + '\n Stemming. ' + file_progress)
        s = PorterStemmer()
        for i in range(len(txt)):
            txt[i] = [s.stem(w) for w in txt[i]]

        # lemmatizer
        logger.info(fname + '\n Lemmatizing. ' + file_progress)
        l = WordNetLemmatizer()
        for i in range(len(txt)):
            txt[i] = [l.lemmatize(w) for w in txt[i]]

        # eliminate most frequent words
        logger.info(fname + '\n Eliminating most frequent words. ' + file_progress)
        all_words = [word for item in txt for word in item]
        fdist = FreqDist(all_words)
        sw = fdist.most_common(int(len(fdist) * THRESHOLD))
        sw = [x[0] for x in sw]
        for i in range(len(txt)):
            txt[i] = [w for w in txt[i] if w not in sw]

        logger.info(fname + '\n Building dictionary. ' + file_progress)

        token_counter = 0

        tmp = []
        original_tmp = ''

        txt_tokenized = []
        original_tokenized = []

        for i in range(len(txt)):

            tmp += txt[i]
            original_tmp += original_txt[i]
            original_tmp += '\t'

            token_counter += len(txt[i])

            if token_counter >= TOKEN_LENGTH:
                token_counter = 0

                txt_tokenized.append(tmp)
                original_tokenized.append(original_tmp)

                tmp = []
                original_tmp = ''

        assert len(original_tokenized) == len(txt_tokenized)

        documents.append({
            'tokens': txt_tokenized,
            'original': original_tokenized,
            'source': fname
        })

    return documents


documents = preprocessing()

# Topic vectors
logger.info('Building topic vectors.')

texts = []
original_texts = []
for dct in documents:
    texts += dct['tokens']
    original_texts += dct['original']

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

hdp_model = HdpModel(corpus, dictionary)
lda_model = hdp_model.suggested_lda_model()

topic_vectors = []

for vec in lda_model[corpus]:
    distribution = [0.0] * 150
    for tup in vec:
        distribution[tup[0]] = tup[1]
    topic_vectors.append(distribution)

# Nearest neighbors
logger.info('Calculating nearest neighbors.')

nearest_neighbors = []
vec = topic_vectors

t = AnnoyIndex(len(vec[0]))  # Length of item vector that will be indexed
for idx, item in enumerate(vec):
    t.add_item(idx, item)
t.build(ann_tree_depth)

for i in range(len(vec)):
    nearest_neighbors.append(t.get_nns_by_item(i, similar_size))


# Build entries
logger.info('Building Elasticsearch input.')


def find_source(index, documents):
    doc_list = []
    for dct in documents:
        doc_list.append(len(dct['tokens']))
    doc_list = np.cumsum(doc_list)
    for idx, item in enumerate(doc_list):
        if item >= index:
            # Erase [6:-4] if you don't want the filename to be trimmed
            return documents[idx]['source'][6:-4]


write_string = ''
entry_list = []
for idx, topic in enumerate(topic_vectors):
    entry = {}
    entry['body'] = original_texts[idx]
    entry['source'] = find_source(idx, documents)
    entry['id'] = idx
    entry['similar'] = nearest_neighbors[idx]
    entry_list.append(entry)

open(vedabase_corpus, 'w+').close()
with open(vedabase_corpus, 'w') as f:
    json.dump(entry_list, f)


def parser():
    for doc in entry_list:
        entry = {}
        entry['_index'] = elastic_index
        entry['_type'] = elastic_type
        entry['_id'] = doc['id']
        entry['body'] = doc['body']
        entry['source'] = doc['source']
        for neighbor_index, neighbor_id in enumerate(doc['similar']):
            entry[neighbor_index] = neighbor_id
        yield entry


# Indexing
logger.info('Calling Elasticsearch bulk API.')

es = Elasticsearch()

es.indices.delete(index=elastic_index, ignore=[400, 404])
es.indices.create(index=elastic_index, ignore=400)
success, _ = bulk(es, parser())

logger.info('Performed %d actions to Elasticsearch database.' % success)
