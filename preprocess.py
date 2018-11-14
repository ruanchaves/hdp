import uuid
import base64
import os
import sys
from dask.bag import read_text
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
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

class ParserController(object):

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


class Document(object):

	def __init__(self, index, fname, byte, lemma, tokens, text):
		self.index = index
		self.fname = fname
		self.bytes = byte
		self.lemma = lemma
		self.tokens = tokens
		self.text = text

class Parser(object):
	
	def __init__(self, nlp, flist, lines ):
		self.dir = directory
		self.nlp = nlp
		self.files = flist
		self.text = [ ( bagline.text, (bagline.index, bagline.fname) ) for bagline in lines ]
		self.doc_list = { x : {} for x in flist }
		self.docs = []

	def run(self,batch_size=10**4):
		for doc in self.nlp.pipe(self.text, as_tuples=True, batch_size=batch_size):
			self.docs.append(doc)

	def parse_worker(self, doc, i):
		doc_object = self.docs[i][0]

		doc_index = self.docs[i][1][0]
		doc_fname = self.docs[i][1][1]
		doc_bytes = base64.encodestring(doc_object.to_bytes())
		doc_lemma = [ token.lemma_.lower() for token in doc_object ]
		doc_lemma = [ x for x in doc_lemma if doc_object.is_stop != True and doc_object.is_punct != True ]
		doc_tokens = doc_object.print_tree(flat=True)
		doc_text = doc_object.text
		fields = (doc_index, doc_fname, doc_bytes, doc_lemma, doc_tokens, doc_text)
		self.docs.append(Document(*fields))

	def parse_pool(self,chunksize=POOL_CHUNK_SIZE):
		with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
			executor.map(parse, [ x for x in range(len(self.docs)) ], chunksize=chunksize )
	
	def __call__(self):
		self.run()
		self.parse_pool()
		return self.doc_list
