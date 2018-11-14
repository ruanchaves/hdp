import json
from session import Preprocessor
from spacy.lang.en import English

filelist = ['/media/ruan/DADOS/vedabase/txt/Varnasrama-dharma.txt']
nlp = English()

pp = Preprocessor(nlp,filelist)
json_object = pp()

with open('model.json','w+') as f:
	json.dump(json_object, f)
