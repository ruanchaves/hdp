from preprocess import ParserController as pcontrol
from preprocess import stopwords

class Snippet(object):
	
	def __init__(self):
		self.index = None
		self.fname = None
		self.start = None
		self.end = None

class HDPtrainer(object):
	
	def __init__(self):
	
	
	def __iter__(self):
		pp = pcontrol()
		sw = stopwords(nlp)
		for 
