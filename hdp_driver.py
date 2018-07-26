import nltk
import re
import string
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import pandas as pd
import csv

#TFIDF
from gensim.models import TfidfModel

#HDP
from gensim.corpora import Dictionary
from gensim.models.hdpmodel import HdpModel

#LDA
from gensim.models import LdaModel

#Plotting
import matplotlib.pyplot as plt

class Process(object):

	def __init__(self,filename):
		with open(filename,'r') as f:
			s = f.read().split('\n')
		self.result = s
	
	def tokenize(self):
		for i,j in enumerate(self.result):
			self.result[i] = nltk.word_tokenize(j)

	def lowercase(self):
		for i,j in enumerate(self.result):
			self.result[i] = [token.lower() for token in j]

	def remove_blanks(self):
		for i,j in enumerate(self.result):
			self.result[i] = [token.strip() for token in j]

	def remove_punctuation(self):
		for i,j in enumerate(self.result):
			pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
			self.result[i] = list(filter(None, [pattern.sub('',token) for token in j]))

	def remove_stopwords(self):
		stopwords = nltk.corpus.stopwords.words('english')
		for i,j in enumerate(self.result):
			self.result[i] = [w for w in j if w not in stopwords]
	
	def stemmer(self,stemmer='porter'):
		if stemmer == 'porter':
			s = PorterStemmer()
		elif stemmer == 'snowball':
			s = SnowBallStemmer("english")
		for i,j in enumerate(self.result):
			self.result[i] = [s.stem(w) for w in j]

	def lemmatizer(self):
		lem = WordNetLemmatizer()
		for i,j in enumerate(self.result):
			self.result[i] = [ lem.lemmatize(w) for w in j]

	def all(self):
		self.tokenize()
		self.lowercase()
		self.remove_blanks()
		self.remove_punctuation()
		self.remove_stopwords()
		self.stemmer()
		self.lemmatizer()

class TFIDF(object):

	def __init__(self,data,threshold=2.5):
		self.data = data
		self.corpus = []
		self.dct = Dictionary()
		self.threshold = threshold
		self.stopwords = []
		self.result = []
	
	def get_stopwords(self):
		for line in self.data:
			self.dct.add_documents([line])
		for line in self.data:
			self.corpus.append(self.dct.doc2bow(line))
		tfidf = TfidfModel(self.corpus)
		for i in range(len(self.corpus)):
			vector = tfidf[self.corpus[i]]
			for j,w in enumerate(vector):
				vector_tfidf = w[0] * w[1]
				word = self.data[i][j]
				item = (vector_tfidf, word)
				self.stopwords.append(item)
		self.stopwords = sorted(self.stopwords, key=lambda x: x[0])[::-1]
	
	def del_stopwords(self):
		percent = self.threshold / 100
		index = int(percent * len(self.stopwords))
		stopwords_trimmed = [ self.stopwords[i][1] for i in range(index) ]
		for i,j in enumerate(self.data):
			self.data[i] = [w for w in j if w not in stopwords_trimmed]
	
	def all(self):
		self.get_stopwords()
		self.del_stopwords()

class HDP(object):

	def __init__(self,seed,stream,threshold=0.008):
		self.seed = seed
		self.stream = stream
		self.threshold = threshold
		
		self.dct = Dictionary()
		self.corpus = []
		self.model = None
		self.topics = None

	def build_hdp(self):
		for line in self.seed:
			if line:
				self.dct.add_documents([line])
		for line in self.stream:
			if line:
				self.corpus.append(self.dct.doc2bow(line))
		self.model = HdpModel(self.corpus,self.dct)
		return self.model

	def update_topics(self):
		alpha = self.model.hdp_to_lda()[0]
		self.topics = []
		for index,item in enumerate(alpha):
			if item > self.threshold:
				self.topics.append(index)
	
	def print_table(self,topn=40,filename='data.csv'):
		result = []
		final_result = []
		for index in self.topics:
			result.append( self.model.print_topic(index,topn=40))
		for i in result:
			temp = i.split('+')
			np_list = []
			for j in temp:
				np_list.append(" ".join(re.findall("[a-zA-Z]+",j)))
			final_result.append(np_list)
		with open(filename,"w") as f:
			wr = csv.writer(f)
			for row in final_result:
				wr.writerow(row)

	def build_topics(self,filename,stopwords=None):
		data = Process(filename)
		data.all()
		data = data.result
		data_bow = []
		for line in data:
			if line:
				data_bow.append(self.dct.doc2bow(line))
		
		final_data_bow = []
		for line in data_bow:
			all_topics = self.model[line]
			all_topics = [x for x in all_topics if x[0] in self.topics]
			if all_topics:
				final_data_bow.append(all_topics)
		return final_data_bow

	def plot(self,topics,filename):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		xaxis = np.arange(151)
		yaxis = [0] * 151
		for item in topics:
			for bow in item:
				yaxis[bow[0]] += bow[1]
		plt.plot(xaxis,yaxis,'bo')
		for xy in zip(xaxis,yaxis):
			if xy[1]:
				ax.annotate('%s' % xy[0], xy=xy, textcoords='data')
		plt.grid()
		plt.savefig(filename)

def preprocess(filename,trim_top=2.5):
	data = Process(filename)
	data.all()
	data = data.result
	tfidf = TFIDF(data,threshold=trim_top)
	tfidf.all()
	return tfidf.data
