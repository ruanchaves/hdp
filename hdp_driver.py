from gensim import models, corpora, similarities
from gensim.models import LdaModel
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models.hdpmodel import HdpModel

from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from scipy.stats import entropy

import gensim

import matplotlib
import matplotlib.pyplot as plt

import nltk
import numpy as np

import pandas as pd

import re

import seaborn as sns

import string
import sys
import time

sns.set_style("darkgrid")

class DF(object):

	def __init__(self,filename,column_list,threshold=0.025):
		self.df = pd.read_csv(filename,sep='\t',usecols=column_list)
		self.df = self.df[self.df['text'].map(type) == str]
		self.df['title'].fillna(value="", inplace=True)
		self.df.dropna(axis=0,inplace=True, subset=['text'])
		self.df = self.df.sample(frac=1.0)
		self.df.reset_index(drop=True,inplace=True)
		self.df.head()

		self.corpus = None
		self.dct = None
		self.stopwords = []
		self.threshold = threshold

	def tokenize(self,text):
			return nltk.word_tokenize(text)

	def lowercase(self,text):
			return [token.lower() for token in text]

	def remove_blanks(self,text):
			return [token.strip() for token in text]

	def remove_punctuation(self,text):
		pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
		return list(filter(None, [pattern.sub('',token) for token in text]))

	def remove_stopwords(self,text):
		sw = nltk.corpus.stopwords.words('english')
		if self.stopwords:
			sw += self.stopwords
		return [w for w in text if w not in sw]
	
	def stemmer(self,text):
		s = PorterStemmer()
		return [s.stem(w) for w in text]

	def lemmatizer(self,text):
		lem = WordNetLemmatizer()
		return [ lem.lemmatize(w) for w in text]

	def clean_all(self,text):
		tokens = self.tokenize(text)
		tokens = self.remove_punctuation(self.remove_blanks(self.lowercase(tokens)))
		tokens = self.lemmatizer(self.stemmer(self.remove_stopwords(tokens)))
		return tokens

	def process(self):
		self.df['tokenized'] = self.df['text'].apply(self.clean_all) + self.df['title'].apply(self.clean_all)

	
	def get_stopwords(self):	
		all_words = [word for item in list(self.df['tokenized']) for word in item]
		fdist = FreqDist(all_words)
		self.stopwords = fdist.most_common(int(len(fdist) * self.threshold))
		self.stopwords = [ x[0] for x in self.stopwords ]

	def apply_stopwords(self):
		self.df['tokenized'] = self.df['tokenized'].apply(self.remove_stopwords)

	def run(self):
		self.process()
		self.get_stopwords()
		self.apply_stopwords()


class HDP(object):

	def __init__(self,corpus,dct,df):
		self.dct = dct
		self.corpus = corpus
		self.model = HdpModel(corpus,dct)
		self.df = df
		self.lda = None
		self.topic_dist = None

	def build_lda(self):
		self.lda = self.model.suggested_lda_model()

	def build_topic_dist(self):
		self.topic_dist = []
		for lst in self.lda[self.corpus]:
			distr = np.array([0.0] * 150)
			for tup in lst:	
				distr[tup[0]] = tup[1]		
			self.topic_dist.append(distr)

	def jensen_shannon(self,query, matrix):
		p = query
		q = matrix
		m = 0.5*(p+q)
		E1 = entropy(p,m)
		E2 = entropy(q,m)
		E = E1 + E2
		return np.sqrt(0.5*E)

	def similarity(self,query,matrix,k=10):
		sims = []
		for index,item in enumerate(matrix):
			sims.append(self.jensen_shannon(query,matrix[index]))
		sims = np.array(sims)
		return sims.argsort()[:k]

	def similarity_query(self,index,k=10,n=2):
		bow = self.dct.doc2bow(self.df.iloc[index,n])
		doc_distribution = np.array([0.0] * 150)
		for tup in self.lda.get_document_topics(bow=bow):
			doc_distribution[tup[0]] = tup[1]
		return self.similarity(doc_distribution,self.topic_dist,k)

def build_data(df_dct,df_corpus):
	dct = Dictionary(df_dct)
	return dct, [dct.doc2bow(doc) for doc in df_corpus]
