import uuid
import base64
import os
import sys
from dask.bag import read_text
from collections import Counter	

def stopwords(nlp, percent):
	words = [ token.text for token in nlp if token.is_stop != True and token.is_punct != True ]
	standard = [ token.text for token in nlp if token.is_stop ]
	word_freq = Counter(words)
	return word_freq.most_common( percent * len(words) ) + standard

class BagLine(object):
	
	def __init__(self, text, index, fname):
		self.text = text
		self.index = index
		self.fname = fname

class FileBag(object):
	
	def __init__(self, f, blocksize=BAG_BLOCK_SIZE):
		self.fname = f
		self.blocks = read_text(f, blocksize=blocksize)
		self.index = 0
		self.block_counter = 0

	def __iter__(self):
		for block in self.blocks:
			lines = block.split('\n')
			for line in lines:
				yield BagLine(line,self.index,self.fname)
				self.index += 1
			self.block_counter += 1	

class FolderReader(object):

	def __init__(self, flist, blocksize=BAG_BLOCK_SIZE):
		self.blocksize = blocksize * ( 1e+6 )
		self.flist = flist
		self.block_list = []
	
	def reset(self):
		self.block_list = [0] * len(self.flist)

	def __iter__(self):
		self.reset()
		baglist = [ FileBag(f, blocksize = self.blocksize) for f in self.flist ]
		for index,bag in enumerate(baglist):
			for line in bag:
				block_list[index] = bag.block_counter
				yield line		

class Controller(object):

	def __init__(self, dirname, nlp, blocksize=BAG_BLOCK_SIZE, blocklimit=BAG_LIMIT):
		self.reader = FolderReader(dirname, blocksize)
		self.nlp = nlp
		self.text = []
		self.blocklimit = blocklimit
		self.flist = os.listdir(dirname)

	def reset(self):
		self.text = []

	def update(self):
		self.flist = os.listdir(dirname)

	def __iter__(self):
		for line in self.reader:
			self.text.append(line)
			if sum(self.reader.block_list) >= self.blocklimit:
				parser = Parser(self.nlp, self.flist, self.text)
				parsed_json = parser()
				yield parsed_json
				self.reader.reset()
				self.reset()

class Parser(object):
	
	def __init__(self, nlp, flist, lines ):
		self.dir = directory
		self.nlp = nlp
		self.files = flist
		self.text = [ ( bagline.text, (bagline.index, bagline.fname) ) for bagline in lines ]
		self.docs = []

	def run(self,batch_size=10**4):
		for doc in self.nlp.pipe(self.text, as_tuples=True, batch_size=batch_size):
			self.docs.append( [ base64.encodestring(doc[0].to_bytes()) , doc[1][0] , doc[1][1] ] )
	
	def __call__(self):
		self.run()
		return self.docs
